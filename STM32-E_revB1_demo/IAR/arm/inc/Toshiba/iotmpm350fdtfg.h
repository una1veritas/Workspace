/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Toshiba TMPM350FDTFG
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 45019 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOTMPM350FDTFG_H
#define __IOTMPM350FDTFG_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMPM350FDTFG SPECIAL FUNCTION REGISTERS
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

/*Test And Set register n TASn (n = 0 to 31)*/
typedef struct {
  __REG32 TASR : 1;
  __REG32      :31;
} __tas_bits;

/*Exclusive Control Interrupt register n EXINTn (n = 0, 1)*/
typedef struct {
  __REG32 EXINTR  : 8;
  __REG32         :24;
} __exint_bits;

/*FNCSELA*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32 FS3  : 2;
  __REG32 FS4  : 2;
  __REG32 FS5  : 2;
  __REG32 FS6  : 2;
  __REG32      :18;
} __fncsela_bits;

/*PDENA*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32 PD3  : 1;
  __REG32 PD4  : 1;
  __REG32 PD5  : 1;
  __REG32 PD6  : 1;
  __REG32      :25;
} __pdena_bits;

/*PUENA*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32 PU3  : 1;
  __REG32 PU4  : 1;
  __REG32 PU5  : 1;
  __REG32 PU6  : 1;
  __REG32      :25;
} __puena_bits;

/*FNCSELB*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32 FS3  : 2;
  __REG32 FS4  : 2;
  __REG32 FS5  : 2;
  __REG32 FS6  : 2;
  __REG32 FS7  : 2;
  __REG32      :16;
} __fncselb_bits;

/*PDENB*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32 PD3  : 1;
  __REG32 PD4  : 1;
  __REG32 PD5  : 1;
  __REG32 PD6  : 1;
  __REG32 PD7  : 1;
  __REG32      :24;
} __pdenb_bits;

/*PUENB*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32 PU3  : 1;
  __REG32 PU4  : 1;
  __REG32 PU5  : 1;
  __REG32 PU6  : 1;
  __REG32 PU7  : 1;
  __REG32      :24;
} __puenb_bits;

/*FNCSELC*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32      :28;
} __fncselc_bits;

/*FNCSELD*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32 FS3  : 2;
  __REG32 FS4  : 2;
  __REG32 FS5  : 2;
  __REG32 FS6  : 2;
  __REG32 FS7  : 2;
  __REG32      :16;
} __fncseld_bits;

/*PDEND*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32 PD3  : 1;
  __REG32 PD4  : 1;
  __REG32 PD5  : 1;
  __REG32 PD6  : 1;
  __REG32 PD7  : 1;
  __REG32      :24;
} __pdend_bits;

/*PUEND*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32 PU3  : 1;
  __REG32 PU4  : 1;
  __REG32 PU5  : 1;
  __REG32 PU6  : 1;
  __REG32 PU7  : 1;
  __REG32      :24;
} __puend_bits;

/*FNCSELE*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32      :26;
} __fncsele_bits;

/*PDENE*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32      :29;
} __pdene_bits;

/*PUENE*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32      :29;
} __puene_bits;

/*FNCSELF*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32 FS3  : 2;
  __REG32 FS4  : 2;
  __REG32 FS5  : 2;
  __REG32      :20;
} __fncself_bits;

/*PDENF*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32 PD3  : 1;
  __REG32 PD4  : 1;
  __REG32 PD5  : 1;
  __REG32      :26;
} __pdenf_bits;

/*PUENF*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32 PU3  : 1;
  __REG32 PU4  : 1;
  __REG32 PU5  : 1;
  __REG32      :26;
} __puenf_bits;

/*FNCSELG*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32      :26;
} __fncselg_bits;

/*PDENG*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32      :29;
} __pdeng_bits;

/*PUENG*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32      :29;
} __pueng_bits;

/*FNCSELH*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32 FS3  : 2;
  __REG32 FS4  : 2;
  __REG32 FS5  : 2;
  __REG32      :20;
} __fncselh_bits;

/*PDENH*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32 PD3  : 1;
  __REG32 PD4  : 1;
  __REG32 PD5  : 1;
  __REG32      :26;
} __pdenh_bits;

/*PUENH*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32 PU3  : 1;
  __REG32 PU4  : 1;
  __REG32 PU5  : 1;
  __REG32      :26;
} __puenh_bits;

/*SEICSCTL*/
typedef struct {
  __REG32 CS0  : 1;
  __REG32 CS1  : 1;
  __REG32 CS2  : 1;
  __REG32      :29;
} __seicsctl_bits;

/*CANM*/
typedef struct {
  __REG32 MOD  : 2;
  __REG32      :30;
} __canm_bits;

/*SIOCKEN*/
typedef struct {
  __REG32 SCK0EN  : 1;
  __REG32 SCK1EN  : 1;
  __REG32     		:30;
} __siocken_bits;

/*MONA*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32 I3   : 1;
  __REG32 I4   : 1;
  __REG32 I5   : 1;
  __REG32 I6   : 1;
  __REG32      :25;
} __mona_bits;

/*OUTA*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32 O3   : 1;
  __REG32 O4   : 1;
  __REG32 O5   : 1;
  __REG32 O6   : 1;
  __REG32      :25;
} __outa_bits;

/*OENA*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32 OE3  : 1;
  __REG32 OE4  : 1;
  __REG32 OE5  : 1;
  __REG32 OE6  : 1;
  __REG32      :25;
} __oena_bits;

/*MONB*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32 I3   : 1;
  __REG32 I4   : 1;
  __REG32 I5   : 1;
  __REG32 I6   : 1;
  __REG32 I7   : 1;
  __REG32      :24;
} __monb_bits;

/*OUTB*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32 O3   : 1;
  __REG32 O4   : 1;
  __REG32 O5   : 1;
  __REG32 O6   : 1;
  __REG32 O7   : 1;
  __REG32      :24;
} __outb_bits;

/*OENB*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32 OE3  : 1;
  __REG32 OE4  : 1;
  __REG32 OE5  : 1;
  __REG32 OE6  : 1;
  __REG32 OE7  : 1;
  __REG32      :24;
} __oenb_bits;

/*MONC*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32      :29;
} __monc_bits;

/*MOND*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32 I3   : 1;
  __REG32 I4   : 1;
  __REG32 I5   : 1;
  __REG32 I6   : 1;
  __REG32 I7   : 1;
  __REG32      :24;
} __mond_bits;

/*OUTD*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32 O3   : 1;
  __REG32 O4   : 1;
  __REG32 O5   : 1;
  __REG32 O6   : 1;
  __REG32 O7   : 1;
  __REG32      :24;
} __outd_bits;

/*OEND*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32 OE3  : 1;
  __REG32 OE4  : 1;
  __REG32 OE5  : 1;
  __REG32 OE6  : 1;
  __REG32 OE7  : 1;
  __REG32      :24;
} __oend_bits;

/*MONE*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32      :29;
} __mone_bits;

/*OUTE*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32      :29;
} __oute_bits;

/*OENE*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32      :29;
} __oene_bits;

/*MONF*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32 I3   : 1;
  __REG32 I4   : 1;
  __REG32 I5   : 1;
  __REG32      :26;
} __monf_bits;

/*OUTF*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32 O3   : 1;
  __REG32 O4   : 1;
  __REG32 O5   : 1;
  __REG32      :26;
} __outf_bits;

/*OENF*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32 OE3  : 1;
  __REG32 OE4  : 1;
  __REG32 OE5  : 1;
  __REG32      :26;
} __oenf_bits;

/*MONG*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32      :29;
} __mong_bits;

/*OUTG*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32      :29;
} __outg_bits;

/*OENG*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32      :29;
} __oeng_bits;

/*MONH*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32 I3   : 1;
  __REG32 I4   : 1;
  __REG32 I5   : 1;
  __REG32      :26;
} __monh_bits;

/*OUTH*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32 O3   : 1;
  __REG32 O4   : 1;
  __REG32 O5   : 1;
  __REG32      :26;
} __outh_bits;

/*OENH*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32 OE3  : 1;
  __REG32 OE4  : 1;
  __REG32 OE5  : 1;
  __REG32      :26;
} __oenh_bits;

/*MONK*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32      :30;
} __monk_bits;

/*TBxxTRUN*/
typedef struct {
  __REG32  TBRUN    : 1;
  __REG32  DIVRUN   : 1;
  __REG32           : 2;
  __REG32  TBPSEN   : 1;
  __REG32           :27;
} __tbtrun_bits;

/*TBxxTCR*/
typedef struct {
  __REG32           : 4;
  __REG32  DIVSEL   : 7;
  __REG32           :21;
} __tbtcr_bits;

/*TBxxCMPxCR*/
typedef struct {
  __REG32  CPMEN	  : 1;
  __REG32  CMPDBEN  : 1;
  __REG32  CNTCLEN  : 1;
  __REG32           :29;
} __tbcmpcr_bits;

/*TBxxCMPDOx*/
typedef struct {
  __REG32  DOR   	  : 1;
  __REG32           :31;
} __tbcmpdo_bits;

/*TBxxCMPSDx*/
typedef struct {
  __REG32  LGTO   	: 1;
  __REG32           :31;
} __tbcmpsd_bits;

/*TBxxCMPOMx*/
typedef struct {
  __REG32  DOM    	: 1;
  __REG32           :31;
} __tbcmpom_bits;

/*DMA Transfer Channel Enable Register (DMACEN)*/
typedef struct {
  __REG32  CEN0     : 1;
  __REG32  CEN1     : 1;
  __REG32  CEN2     : 1;
  __REG32  CEN3     : 1;
  __REG32  CEN4     : 1;
  __REG32  CEN5     : 1;
  __REG32  CEN6     : 1;
  __REG32  CEN7     : 1;
  __REG32  CEN8     : 1;
  __REG32  CEN9     : 1;
  __REG32  CEN10    : 1;
  __REG32  CEN11    : 1;
  __REG32  CEN12    : 1;
  __REG32  CEN13    : 1;
  __REG32  CEN14    : 1;
  __REG32  CEN15    : 1;
  __REG32  CEN16    : 1;
  __REG32  CEN17    : 1;
  __REG32  CEN18    : 1;
  __REG32  CEN19    : 1;
  __REG32  CEN20    : 1;
  __REG32  CEN21    : 1;
  __REG32  CEN22    : 1;
  __REG32  CEN23    : 1;
  __REG32  CEN24    : 1;
  __REG32  CEN25    : 1;
  __REG32  CEN26    : 1;
  __REG32  CEN27    : 1;
  __REG32  CEN28    : 1;
  __REG32  CEN29    : 1;
  __REG32  CEN30    : 1;
  __REG32  CEN31    : 1;
} __dmacen_bits;

/*DMA Transfer Request Register (DMAREQ)*/
typedef struct {
  __REG32  REQ0     : 1;
  __REG32  REQ1     : 1;
  __REG32  REQ2     : 1;
  __REG32  REQ3     : 1;
  __REG32  REQ4     : 1;
  __REG32  REQ5     : 1;
  __REG32  REQ6     : 1;
  __REG32  REQ7     : 1;
  __REG32  REQ8     : 1;
  __REG32  REQ9     : 1;
  __REG32  REQ10    : 1;
  __REG32  REQ11    : 1;
  __REG32  REQ12    : 1;
  __REG32  REQ13    : 1;
  __REG32  REQ14    : 1;
  __REG32  REQ15    : 1;
  __REG32  REQ16    : 1;
  __REG32  REQ17    : 1;
  __REG32  REQ18    : 1;
  __REG32  REQ19    : 1;
  __REG32  REQ20    : 1;
  __REG32  REQ21    : 1;
  __REG32  REQ22    : 1;
  __REG32  REQ23    : 1;
  __REG32  REQ24    : 1;
  __REG32  REQ25    : 1;
  __REG32  REQ26    : 1;
  __REG32  REQ27    : 1;
  __REG32  REQ28    : 1;
  __REG32  REQ29    : 1;
  __REG32  REQ30    : 1;
  __REG32  REQ31    : 1;
} __dmareq_bits;

/*DMA Transfer Suspend Register (DMASUS)*/
typedef struct {
  __REG32  SUS0     : 1;
  __REG32  SUS1     : 1;
  __REG32  SUS2     : 1;
  __REG32  SUS3     : 1;
  __REG32  SUS4     : 1;
  __REG32  SUS5     : 1;
  __REG32  SUS6     : 1;
  __REG32  SUS7     : 1;
  __REG32  SUS8     : 1;
  __REG32  SUS9     : 1;
  __REG32  SUS10    : 1;
  __REG32  SUS11    : 1;
  __REG32  SUS12    : 1;
  __REG32  SUS13    : 1;
  __REG32  SUS14    : 1;
  __REG32  SUS15    : 1;
  __REG32  SUS16    : 1;
  __REG32  SUS17    : 1;
  __REG32  SUS18    : 1;
  __REG32  SUS19    : 1;
  __REG32  SUS20    : 1;
  __REG32  SUS21    : 1;
  __REG32  SUS22    : 1;
  __REG32  SUS23    : 1;
  __REG32  SUS24    : 1;
  __REG32  SUS25    : 1;
  __REG32  SUS26    : 1;
  __REG32  SUS27    : 1;
  __REG32  SUS28    : 1;
  __REG32  SUS29    : 1;
  __REG32  SUS30    : 1;
  __REG32  SUS31    : 1;
} __dmasus_bits;

/*DMA Transfer Active Register (DMAACT)*/
typedef struct {
  __REG32  ACT0     : 1;
  __REG32  ACT1     : 1;
  __REG32  ACT2     : 1;
  __REG32  ACT3     : 1;
  __REG32  ACT4     : 1;
  __REG32  ACT5     : 1;
  __REG32  ACT6     : 1;
  __REG32  ACT7     : 1;
  __REG32  ACT8     : 1;
  __REG32  ACT9     : 1;
  __REG32  ACT10    : 1;
  __REG32  ACT11    : 1;
  __REG32  ACT12    : 1;
  __REG32  ACT13    : 1;
  __REG32  ACT14    : 1;
  __REG32  ACT15    : 1;
  __REG32  ACT16    : 1;
  __REG32  ACT17    : 1;
  __REG32  ACT18    : 1;
  __REG32  ACT19    : 1;
  __REG32  ACT20    : 1;
  __REG32  ACT21    : 1;
  __REG32  ACT22    : 1;
  __REG32  ACT23    : 1;
  __REG32  ACT24    : 1;
  __REG32  ACT25    : 1;
  __REG32  ACT26    : 1;
  __REG32  ACT27    : 1;
  __REG32  ACT28    : 1;
  __REG32  ACT29    : 1;
  __REG32  ACT30    : 1;
  __REG32  ACT31    : 1;
} __dmaact_bits;

/*DMA Transfer End Register (DMAEND)*/
typedef struct {
  __REG32  END0     : 1;
  __REG32  END1     : 1;
  __REG32  END2     : 1;
  __REG32  END3     : 1;
  __REG32  END4     : 1;
  __REG32  END5     : 1;
  __REG32  END6     : 1;
  __REG32  END7     : 1;
  __REG32  END8     : 1;
  __REG32  END9     : 1;
  __REG32  END10    : 1;
  __REG32  END11    : 1;
  __REG32  END12    : 1;
  __REG32  END13    : 1;
  __REG32  END14    : 1;
  __REG32  END15    : 1;
  __REG32  END16    : 1;
  __REG32  END17    : 1;
  __REG32  END18    : 1;
  __REG32  END19    : 1;
  __REG32  END20    : 1;
  __REG32  END21    : 1;
  __REG32  END22    : 1;
  __REG32  END23    : 1;
  __REG32  END24    : 1;
  __REG32  END25    : 1;
  __REG32  END26    : 1;
  __REG32  END27    : 1;
  __REG32  END28    : 1;
  __REG32  END29    : 1;
  __REG32  END30    : 1;
  __REG32  END31    : 1;
} __dmaend_bits;

/*DMA Transfer Priority Set Register (DMAPRI)*/
typedef struct {
  __REG32  PRI0     : 1;
  __REG32  PRI1     : 1;
  __REG32  PRI2     : 1;
  __REG32  PRI3     : 1;
  __REG32  PRI4     : 1;
  __REG32  PRI5     : 1;
  __REG32  PRI6     : 1;
  __REG32  PRI7     : 1;
  __REG32  PRI8     : 1;
  __REG32  PRI9     : 1;
  __REG32  PRI10    : 1;
  __REG32  PRI11    : 1;
  __REG32  PRI12    : 1;
  __REG32  PRI13    : 1;
  __REG32  PRI14    : 1;
  __REG32  PRI15    : 1;
  __REG32  PRI16    : 1;
  __REG32  PRI17    : 1;
  __REG32  PRI18    : 1;
  __REG32  PRI19    : 1;
  __REG32  PRI20    : 1;
  __REG32  PRI21    : 1;
  __REG32  PRI22    : 1;
  __REG32  PRI23    : 1;
  __REG32  PRI24    : 1;
  __REG32  PRI25    : 1;
  __REG32  PRI26    : 1;
  __REG32  PRI27    : 1;
  __REG32  PRI28    : 1;
  __REG32  PRI29    : 1;
  __REG32  PRI30    : 1;
  __REG32  PRI31    : 1;
} __dmapri_bits;

/*DMA Transfer Priority Set Register (DMAPRI)*/
typedef struct {
  __REG32  ENE0     : 1;
  __REG32  ENE1     : 1;
  __REG32  ENE2     : 1;
  __REG32  ENE3     : 1;
  __REG32  ENE4     : 1;
  __REG32  ENE5     : 1;
  __REG32  ENE6     : 1;
  __REG32  ENE7     : 1;
  __REG32  ENE8     : 1;
  __REG32  ENE9     : 1;
  __REG32  ENE10    : 1;
  __REG32  ENE11    : 1;
  __REG32  ENE12    : 1;
  __REG32  ENE13    : 1;
  __REG32  ENE14    : 1;
  __REG32  ENE15    : 1;
  __REG32  ENE16    : 1;
  __REG32  ENE17    : 1;
  __REG32  ENE18    : 1;
  __REG32  ENE19    : 1;
  __REG32  ENE20    : 1;
  __REG32  ENE21    : 1;
  __REG32  ENE22    : 1;
  __REG32  ENE23    : 1;
  __REG32  ENE24    : 1;
  __REG32  ENE25    : 1;
  __REG32  ENE26    : 1;
  __REG32  ENE27    : 1;
  __REG32  ENE28    : 1;
  __REG32  ENE29    : 1;
  __REG32  ENE30    : 1;
  __REG32  ENE31    : 1;
} __dmaene_bits;

/*DMA Transfer Execution Channel Number Register (DMACHN)*/
typedef struct {
  __REG32  CHNUM    : 5;
  __REG32           :27;
} __dmachn_bits;

/*DMA Transfer Execution Channel Number Register (DMACHN)*/
typedef struct {
  __REG32  TTYPE    : 2;
  __REG32           : 6;
  __REG32  USIZE    : 3;
  __REG32           : 5;
  __REG32  UMODE    : 1;
  __REG32           : 7;
  __REG32  DMODE    : 1;
  __REG32           : 7;
} __dmaxftyp_bits;

/*DMA Transfer Size Register (DMAXFSIZ)*/
typedef struct {
  __REG32  XFSIZ    :20;
  __REG32           :12;
} __dmaxfsiz_bits;

/*DMA Transfer Size Register (DMAXFSIZ)*/
typedef struct {
  __REG32  DSNUM    : 8;
  __REG32           :24;
} __dmadsnum_bits;

/*DMA Transfer Request Signal Level Control Register (DMALRQ)*/
typedef struct {
  __REG32  LRQ0     : 1;
  __REG32  LRQ1     : 1;
  __REG32  LRQ2     : 1;
  __REG32  LRQ3     : 1;
  __REG32  LRQ4     : 1;
  __REG32  LRQ5     : 1;
  __REG32  LRQ6     : 1;
  __REG32  LRQ7     : 1;
  __REG32  LRQ8     : 1;
  __REG32  LRQ9     : 1;
  __REG32  LRQ10    : 1;
  __REG32  LRQ11    : 1;
  __REG32  LRQ12    : 1;
  __REG32  LRQ13    : 1;
  __REG32  LRQ14    : 1;
  __REG32  LRQ15    : 1;
  __REG32  LRQ16    : 1;
  __REG32  LRQ17    : 1;
  __REG32  LRQ18    : 1;
  __REG32  LRQ19    : 1;
  __REG32  LRQ20    : 1;
  __REG32  LRQ21    : 1;
  __REG32  LRQ22    : 1;
  __REG32  LRQ23    : 1;
  __REG32  LRQ24    : 1;
  __REG32  LRQ25    : 1;
  __REG32  LRQ26    : 1;
  __REG32  LRQ27    : 1;
  __REG32  LRQ28    : 1;
  __REG32  LRQ29    : 1;
  __REG32  LRQ30    : 1;
  __REG32  LRQ31    : 1;
} __dmalrq_bits;

/*DMA Transfer Mask Register (DMAMSK)*/
typedef struct {
  __REG32  MSK0     : 1;
  __REG32  MSK1     : 1;
  __REG32  MSK2     : 1;
  __REG32  MSK3     : 1;
  __REG32  MSK4     : 1;
  __REG32  MSK5     : 1;
  __REG32  MSK6     : 1;
  __REG32  MSK7     : 1;
  __REG32  MSK8     : 1;
  __REG32  MSK9     : 1;
  __REG32  MSK10    : 1;
  __REG32  MSK11    : 1;
  __REG32  MSK12    : 1;
  __REG32  MSK13    : 1;
  __REG32  MSK14    : 1;
  __REG32  MSK15    : 1;
  __REG32  MSK16    : 1;
  __REG32  MSK17    : 1;
  __REG32  MSK18    : 1;
  __REG32  MSK19    : 1;
  __REG32  MSK20    : 1;
  __REG32  MSK21    : 1;
  __REG32  MSK22    : 1;
  __REG32  MSK23    : 1;
  __REG32  MSK24    : 1;
  __REG32  MSK25    : 1;
  __REG32  MSK26    : 1;
  __REG32  MSK27    : 1;
  __REG32  MSK28    : 1;
  __REG32  MSK29    : 1;
  __REG32  MSK30    : 1;
  __REG32  MSK31    : 1;
} __dmamsk_bits;

/*TMR TBT RUN Register (TMR_TBTRUN)*/
typedef struct {
  __REG32  TBTRUN   : 1;
  __REG32  DIVRUN   : 1;
  __REG32           : 2;
  __REG32  TBTPSEN  : 1;
  __REG32           :27;
} __tmr_tbtrun_bits;

/*TMR TBT Control Register (TMR_TBTCR)*/
typedef struct {
  __REG32           : 4;
  __REG32  DIVSEL   : 7;
  __REG32           :21;
} __tmr_tbtcr_bits;

/*TMR Capture Control Register (TMR_CAPCR)*/
typedef struct {
  __REG32  CAPNF0   : 1;
  __REG32  CAPNF1   : 1;
  __REG32  CAPNF2   : 1;
  __REG32  CAPNF3   : 1;
  __REG32  CAPNF4   : 1;
  __REG32  CAPNF5   : 1;
  __REG32  CAPNF6   : 1;
  __REG32  CAPNF7   : 1;
  __REG32           :24;
} __tmr_capcr_bits;

/*Compare Control Register 0/1 (TMR_CMPxCR)*/
typedef struct {
  __REG32  CMPEN    : 1;
  __REG32  CMPDBEN  : 1;
  __REG32  CNTCLEN  : 1;
  __REG32           :29;
} __tmr_cmpcr_bits;

/*PMD0MDEN*/
typedef struct {
  __REG32  PWMEN    : 1;
  __REG32           :31;
} __pmdmden_bits;

/*PMD0PORTMD*/
typedef struct {
  __REG32  PORTMD   : 2;
  __REG32           :30;
} __pmdportmd_bits;

/*PMD0MODESEL*/
typedef struct {
  __REG32  MDSEL    : 1;
  __REG32           :31;
} __pmdmodesel_bits;

/*PMD0MDCR*/
typedef struct {
  __REG32  PWMMD    : 1;
  __REG32  INTPRD   : 2;
  __REG32  PINT     : 1;
  __REG32  DTYMD    : 1;
  __REG32  SYNTMD   : 1;
  __REG32  PWMCK    : 1;
  __REG32           :25;
} __pmdmdpr_bits;

/*PMD0CNTSTA*/
typedef struct {
  __REG32  UPDWN    : 1;
  __REG32           :31;
} __pmdcntsta_bits;

/*PMD0MDCNT*/
typedef struct {
  __REG32  MDCNT    :16;
  __REG32           :16;
} __pmdmdcnt_bits;

/*PMD0MDPRD*/
typedef struct {
  __REG32  MDPRD    :16;
  __REG32           :16;
} __pmdmdprd_bits;

/*PMD0CMPU*/
typedef struct {
  __REG32  CMPU     :16;
  __REG32           :16;
} __pmdcmpu_bits;

/*PMD0CMPV*/
typedef struct {
  __REG32  CMPV     :16;
  __REG32           :16;
} __pmdcmpv_bits;

/*PMD0CMPW*/
typedef struct {
  __REG32  CMPW     :16;
  __REG32           :16;
} __pmdcmpw_bits;

/*PMD0MDPOT*/
typedef struct {
  __REG32  PSYNCS   : 2;
  __REG32  POLL    	: 1;
  __REG32  POLH 	  : 1;
  __REG32           :28;
} __pmdmdpot_bits;

/*PMD0MDOUT*/
typedef struct {
  __REG32  UOC      : 2;
  __REG32  VOC    	: 2;
  __REG32  WOC  	  : 2;
  __REG32           :26;
} __pmdmdout_bits;

/*PMD0EMGREL*/
typedef struct {
  __REG32  EMGREL   : 8;
  __REG32           :24;
} __pmdemgrel_bits;

/*PMD0EMGCR*/
typedef struct {
  __REG32  EMGEN    : 1;
  __REG32  EMGRS    : 1;
  __REG32  EMGISEL  : 1;
  __REG32  EMGMD    : 2;
  __REG32  INHEN    : 1;
  __REG32           :26;
} __pmdemgcr_bits;

/*PMD0EMGSTA*/
typedef struct {
  __REG32  EMGST    : 1;
  __REG32  EMGI     : 1;
  __REG32           :30;
} __pmdemgsta_bits;

/*PMD0DTR*/
typedef struct {
  __REG32  DTR      : 8;
  __REG32           :24;
} __pmddtr_bits;

/*PMD0TRGCMPx*/
typedef struct {
  __REG32  TRGCMP   :16;
  __REG32           :16;
} __pmdtrgcmp_bits;

/*PMD0TRGCR*/
typedef struct {
  __REG32  TRG0MD   : 3;
  __REG32  TRG0BE   : 1;
  __REG32  TRG1MD   : 3;
  __REG32  TRG1BE   : 1;
  __REG32  TRG2MD   : 3;
  __REG32  TRG2BE   : 1;
  __REG32  TRG3MD   : 3;
  __REG32  TRG3BE   : 1;
  __REG32           :16;
} __pmdtrgcr_bits;

/*PMD0TRGMD*/
typedef struct {
  __REG32  EMGTGE   : 1;
  __REG32  TRGOUT   : 1;
  __REG32           :30;
} __pmdtrgmd_bits;

/*PMD0TRGSEL*/
typedef struct {
  __REG32  TRGSEL   : 3;
  __REG32           :29;
} __pmdtrgsel_bits;

/*PWM Sync Control Register (PWMSYNCRUN)*/
typedef struct {
  __REG32  R0  		  : 1;
  __REG32  PR0  		: 1;
  __REG32  R1  		  : 1;
  __REG32  PR1  		: 1;
  __REG32  R2  		  : 1;
  __REG32  PR2  		: 1;
  __REG32  					:26;
} __pwmsyncrun_bits;

/*PWM RUN Register PWMn_RUN (n=0 to F)*/
typedef struct {
  __REG32  TBRUN    : 1;
  __REG32           : 1;
  __REG32  TBPRUN   : 1;
  __REG32           : 1;
  __REG32  TBPSEN   : 1;
  __REG32           :27;
} __pwm_run_bits;

/*PWM Control Register PWMn_CR (n=0 to F)*/
typedef struct {
  __REG32  TBWBUF   : 1;
  __REG32           :31;
} __pwm_cr_bits;

/*PWM Mode Register PWMn_MOD (n=0 to F)*/
typedef struct {
  __REG32  DIVSEL   : 3;
  __REG32           :29;
} __pwm_mod_bits;

/*PWM Output Polarity Register PWMn_OUTCTRL (n=0 to F)*/
typedef struct {
  __REG32  TBACT    : 1;
  __REG32           :31;
} __pwm_outctrl_bits;

/*PWM Period Register PWMn_PRICMP (n=0 to F)*/
typedef struct {
  __REG32  TBRG0    :24;
  __REG32           : 8;
} __pwm_rg0_bits;

/*PWM Period Register PWMn_DUTYCMP (n=0 to F)*/
typedef struct {
  __REG32  TBRG1    :24;
  __REG32           : 8;
} __pwm_rg1_bits;

/*PWM Counter Register PWMn_CNT (n=0 to F)*/
typedef struct {
  __REG32  CNT      :24;
  __REG32           : 8;
} __pwm_cnt_bits;

/*SCxEN*/
typedef struct {
  __REG8 SIOE     : 1;
  __REG8          : 7;
} __scen_bits;

/*SCxCR*/
typedef struct {
  __REG8 IOC      : 1;
  __REG8 SCLKS    : 1;
  __REG8 FERR     : 1;
  __REG8 PERR     : 1;
  __REG8 OERR     : 1;
  __REG8 PE       : 1;
  __REG8 EVEN     : 1;
  __REG8 RB9      : 1;
} __sccr_bits;

/*SCxMOD0*/
typedef struct {
  __REG8 SC       : 2;
  __REG8 SM       : 2;
  __REG8 WU       : 1;
  __REG8 RXE      : 1;
  __REG8 CTSE     : 1;
  __REG8 TB8      : 1;
} __scmod0_bits;

/*SCxMOD1*/
typedef struct {
  __REG8          : 1;
  __REG8 SINT     : 3;
  __REG8 TXE      : 1;
  __REG8 FDPX     : 2;
  __REG8 I2S0     : 1;
} __scmod1_bits;

/*SCxMOD2*/
typedef struct {
  __REG8 SWRST 	  : 2;
  __REG8 WBUF 	  : 1;
  __REG8 DRCHG    : 1;
  __REG8 SBLEN    : 1;
  __REG8 TXRUN    : 1;
  __REG8 RBFLL    : 1;
  __REG8 TBEMP    : 1;
} __scmod2_bits;

/*SC0BRCR*/
typedef struct {
  __REG8 RB0S		  : 4;
  __REG8 RB0CK    : 2;
  __REG8 RB0ADE   : 1;
  __REG8 			    : 1;
} __scbrcr_bits;

/*SC0BRADD*/
typedef struct {
  __REG8 RB0K		  : 4;
  __REG8 			    : 4;
} __scbradd_bits;

/*SC0FCNF*/
typedef struct {
  __REG8 CNFG		  : 1;
  __REG8 RXTXCNT  : 1;
  __REG8 RFIE		  : 1;
  __REG8 TFIE		  : 1;
  __REG8 RFST		  : 1;
  __REG8 			    : 3;
} __scfcnf_bits;

/*SC0RFC*/
typedef struct {
  __REG8 RIL 		  : 2;
  __REG8 				  : 4;
  __REG8 RFIS		  : 1;
  __REG8 RFCS		  : 1;
} __scrfc_bits;

/*SC0TFC*/
typedef struct {
  __REG8 TIL 		  : 2;
  __REG8 				  : 4;
  __REG8 TFIS		  : 1;
  __REG8 TFCS		  : 1;
} __sctfc_bits;

/*SC0RST*/
typedef struct {
  __REG8 RLVL		  : 3;
  __REG8 				  : 4;
  __REG8 ROR 		  : 1;
} __scrst_bits;

/*SC0TST*/
typedef struct {
  __REG8 TLVL		  : 3;
  __REG8 				  : 4;
  __REG8 TUR 		  : 1;
} __sctst_bits;

/*SSPCR0 (SSP Control register 0)*/
typedef struct {
  __REG32 DSS     : 4;
  __REG32 FRF     : 2;
  __REG32 SPO     : 1;
  __REG32 SPH     : 1;
  __REG32 SCR     : 8;
  __REG32         :16;
} __sspcr0_bits;

/*SSPCR1 (SSP Control register 1)*/
typedef struct {
  __REG32 LBM     : 1;
  __REG32 SSE     : 1;
  __REG32 MS      : 1;
  __REG32 SOD     : 1;
  __REG32         :28;
} __sspcr1_bits;

/*SSPDR (SSP Data register)*/
typedef struct {
  __REG32 DATA    :16;
  __REG32         :16;
} __sspdr_bits;

/*SSPSR (SSP Status register)*/
typedef struct {
  __REG32 TFE     : 1;
  __REG32 TNF     : 1;
  __REG32 RNE     : 1;
  __REG32 RFF     : 1;
  __REG32 BSY     : 1;
  __REG32         :27;
} __sspsr_bits;

/*SSPCPSR (SSP Clock prescale register)*/
typedef struct {
  __REG32 CPSDVSR : 8;
  __REG32         :24;
} __sspcpsr_bits;

/*SSPIMSC (SSP Interrupt mask set and clear register)*/
typedef struct {
  __REG32 RORIM   : 1;
  __REG32 RTIM    : 1;
  __REG32 RXIM    : 1;
  __REG32 TXIM    : 1;
  __REG32         :28;
} __sspimsc_bits;

/*SSPRIS (SSP Raw interrupt status register)*/
typedef struct {
  __REG32 RORRIS  : 1;
  __REG32 RTRIS   : 1;
  __REG32 RXRIS   : 1;
  __REG32 TXRIS   : 1;
  __REG32         :28;
} __sspris_bits;

/*SSPMIS (SSP Masked interrupt status register)*/
typedef struct {
  __REG32 RORMIS  : 1;
  __REG32 RTMIS   : 1;
  __REG32 RXMIS   : 1;
  __REG32 TXMIS   : 1;
  __REG32         :28;
} __sspmis_bits;

/*SSPICR (SSP Interrupt clear register)*/
typedef struct {
  __REG32 RORIC   : 1;
  __REG32 RTIC    : 1;
  __REG32         :30;
} __sspicr_bits;

/*SSPDMACR (SSP DMA control register*/
typedef struct {
  __REG32 RXDMAE  : 1;
  __REG32 TXDMAE  : 1;
  __REG32         :30;
} __sspdmacr_bits;

/* CAN Message Identifier (ID0 .. ID3) */
typedef struct {
  __REG32  EXT_ID         :18;
  __REG32  STD_ID         :11;
  __REG32  RFH            : 1;
  __REG32  GAME           : 1;
  __REG32  IDE            : 1;
} __canmbid_bits;

/* CAN Time Stamp Value (TSV)/Message Control Field (MCF) */
typedef struct {
  __REG32  DLC            : 4;
  __REG32  RTR            : 1;
  __REG32                 :11;
  __REG32  TSV            :16;
} __canmbtcr_bits;

/* CAN Mailbox Configuration Register (MC) */
typedef struct {
  __REG32  MC0            : 1;
  __REG32  MC1            : 1;
  __REG32  MC2            : 1;
  __REG32  MC3            : 1;
  __REG32  MC4            : 1;
  __REG32  MC5            : 1;
  __REG32  MC6            : 1;
  __REG32  MC7            : 1;
  __REG32  MC8            : 1;
  __REG32  MC9            : 1;
  __REG32  MC10           : 1;
  __REG32  MC11           : 1;
  __REG32  MC12           : 1;
  __REG32  MC13           : 1;
  __REG32  MC14           : 1;
  __REG32  MC15           : 1;
  __REG32  MC16           : 1;
  __REG32  MC17           : 1;
  __REG32  MC18           : 1;
  __REG32  MC19           : 1;
  __REG32  MC20           : 1;
  __REG32  MC21           : 1;
  __REG32  MC22           : 1;
  __REG32  MC23           : 1;
  __REG32  MC24           : 1;
  __REG32  MC25           : 1;
  __REG32  MC26           : 1;
  __REG32  MC27           : 1;
  __REG32  MC28           : 1;
  __REG32  MC29           : 1;
  __REG32  MC30           : 1;
  __REG32  MC31           : 1;
} __canmc_bits;

/* CAN Mailbox Direction Register (MD) */
typedef struct {
  __REG32  MD0            : 1;
  __REG32  MD1            : 1;
  __REG32  MD2            : 1;
  __REG32  MD3            : 1;
  __REG32  MD4            : 1;
  __REG32  MD5            : 1;
  __REG32  MD6            : 1;
  __REG32  MD7            : 1;
  __REG32  MD8            : 1;
  __REG32  MD9            : 1;
  __REG32  MD10           : 1;
  __REG32  MD11           : 1;
  __REG32  MD12           : 1;
  __REG32  MD13           : 1;
  __REG32  MD14           : 1;
  __REG32  MD15           : 1;
  __REG32  MD16           : 1;
  __REG32  MD17           : 1;
  __REG32  MD18           : 1;
  __REG32  MD19           : 1;
  __REG32  MD20           : 1;
  __REG32  MD21           : 1;
  __REG32  MD22           : 1;
  __REG32  MD23           : 1;
  __REG32  MD24           : 1;
  __REG32  MD25           : 1;
  __REG32  MD26           : 1;
  __REG32  MD27           : 1;
  __REG32  MD28           : 1;
  __REG32  MD29           : 1;
  __REG32  MD30           : 1;
  __REG32  MD31           : 1;
} __canmd_bits;

/* CAN Transmission Request Set Register (TRS) */
typedef struct {
  __REG32  TRS0            : 1;
  __REG32  TRS1            : 1;
  __REG32  TRS2            : 1;
  __REG32  TRS3            : 1;
  __REG32  TRS4            : 1;
  __REG32  TRS5            : 1;
  __REG32  TRS6            : 1;
  __REG32  TRS7            : 1;
  __REG32  TRS8            : 1;
  __REG32  TRS9            : 1;
  __REG32  TRS10           : 1;
  __REG32  TRS11           : 1;
  __REG32  TRS12           : 1;
  __REG32  TRS13           : 1;
  __REG32  TRS14           : 1;
  __REG32  TRS15           : 1;
  __REG32  TRS16           : 1;
  __REG32  TRS17           : 1;
  __REG32  TRS18           : 1;
  __REG32  TRS19           : 1;
  __REG32  TRS20           : 1;
  __REG32  TRS21           : 1;
  __REG32  TRS22           : 1;
  __REG32  TRS23           : 1;
  __REG32  TRS24           : 1;
  __REG32  TRS25           : 1;
  __REG32  TRS26           : 1;
  __REG32  TRS27           : 1;
  __REG32  TRS28           : 1;
  __REG32  TRS29           : 1;
  __REG32  TRS30           : 1;
  __REG32                  : 1;
} __cantrs_bits;

/* CAN Transmission Request Reset Register (TRR) */
typedef struct {
  __REG32  TRR0            : 1;
  __REG32  TRR1            : 1;
  __REG32  TRR2            : 1;
  __REG32  TRR3            : 1;
  __REG32  TRR4            : 1;
  __REG32  TRR5            : 1;
  __REG32  TRR6            : 1;
  __REG32  TRR7            : 1;
  __REG32  TRR8            : 1;
  __REG32  TRR9            : 1;
  __REG32  TRR10           : 1;
  __REG32  TRR11           : 1;
  __REG32  TRR12           : 1;
  __REG32  TRR13           : 1;
  __REG32  TRR14           : 1;
  __REG32  TRR15           : 1;
  __REG32  TRR16           : 1;
  __REG32  TRR17           : 1;
  __REG32  TRR18           : 1;
  __REG32  TRR19           : 1;
  __REG32  TRR20           : 1;
  __REG32  TRR21           : 1;
  __REG32  TRR22           : 1;
  __REG32  TRR23           : 1;
  __REG32  TRR24           : 1;
  __REG32  TRR25           : 1;
  __REG32  TRR26           : 1;
  __REG32  TRR27           : 1;
  __REG32  TRR28           : 1;
  __REG32  TRR29           : 1;
  __REG32  TRR30           : 1;
  __REG32                  : 1;
} __cantrr_bits;

/* CAN Transmission Acknowledge Register (TA) */
typedef struct {
  __REG32  TA0            : 1;
  __REG32  TA1            : 1;
  __REG32  TA2            : 1;
  __REG32  TA3            : 1;
  __REG32  TA4            : 1;
  __REG32  TA5            : 1;
  __REG32  TA6            : 1;
  __REG32  TA7            : 1;
  __REG32  TA8            : 1;
  __REG32  TA9            : 1;
  __REG32  TA10           : 1;
  __REG32  TA11           : 1;
  __REG32  TA12           : 1;
  __REG32  TA13           : 1;
  __REG32  TA14           : 1;
  __REG32  TA15           : 1;
  __REG32  TA16           : 1;
  __REG32  TA17           : 1;
  __REG32  TA18           : 1;
  __REG32  TA19           : 1;
  __REG32  TA20           : 1;
  __REG32  TA21           : 1;
  __REG32  TA22           : 1;
  __REG32  TA23           : 1;
  __REG32  TA24           : 1;
  __REG32  TA25           : 1;
  __REG32  TA26           : 1;
  __REG32  TA27           : 1;
  __REG32  TA28           : 1;
  __REG32  TA29           : 1;
  __REG32  TA30           : 1;
  __REG32                 : 1;
} __canta_bits;

/* CAN Abort Acknowledge Register (AA) */
typedef struct {
  __REG32  AA0            : 1;
  __REG32  AA1            : 1;
  __REG32  AA2            : 1;
  __REG32  AA3            : 1;
  __REG32  AA4            : 1;
  __REG32  AA5            : 1;
  __REG32  AA6            : 1;
  __REG32  AA7            : 1;
  __REG32  AA8            : 1;
  __REG32  AA9            : 1;
  __REG32  AA10           : 1;
  __REG32  AA11           : 1;
  __REG32  AA12           : 1;
  __REG32  AA13           : 1;
  __REG32  AA14           : 1;
  __REG32  AA15           : 1;
  __REG32  AA16           : 1;
  __REG32  AA17           : 1;
  __REG32  AA18           : 1;
  __REG32  AA19           : 1;
  __REG32  AA20           : 1;
  __REG32  AA21           : 1;
  __REG32  AA22           : 1;
  __REG32  AA23           : 1;
  __REG32  AA24           : 1;
  __REG32  AA25           : 1;
  __REG32  AA26           : 1;
  __REG32  AA27           : 1;
  __REG32  AA28           : 1;
  __REG32  AA29           : 1;
  __REG32  AA30           : 1;
  __REG32                 : 1;
} __canaa_bits;

/* CAN Change Data Request (CDR) */
typedef struct {
  __REG32  CDR0            : 1;
  __REG32  CDR1            : 1;
  __REG32  CDR2            : 1;
  __REG32  CDR3            : 1;
  __REG32  CDR4            : 1;
  __REG32  CDR5            : 1;
  __REG32  CDR6            : 1;
  __REG32  CDR7            : 1;
  __REG32  CDR8            : 1;
  __REG32  CDR9            : 1;
  __REG32  CDR10           : 1;
  __REG32  CDR11           : 1;
  __REG32  CDR12           : 1;
  __REG32  CDR13           : 1;
  __REG32  CDR14           : 1;
  __REG32  CDR15           : 1;
  __REG32  CDR16           : 1;
  __REG32  CDR17           : 1;
  __REG32  CDR18           : 1;
  __REG32  CDR19           : 1;
  __REG32  CDR20           : 1;
  __REG32  CDR21           : 1;
  __REG32  CDR22           : 1;
  __REG32  CDR23           : 1;
  __REG32  CDR24           : 1;
  __REG32  CDR25           : 1;
  __REG32  CDR26           : 1;
  __REG32  CDR27           : 1;
  __REG32  CDR28           : 1;
  __REG32  CDR29           : 1;
  __REG32  CDR30           : 1;
  __REG32                 : 1;
} __cancdr_bits;

/* CAN Receive Message Pending Register (RMP) */
typedef struct {
  __REG32  RMP0            : 1;
  __REG32  RMP1            : 1;
  __REG32  RMP2            : 1;
  __REG32  RMP3            : 1;
  __REG32  RMP4            : 1;
  __REG32  RMP5            : 1;
  __REG32  RMP6            : 1;
  __REG32  RMP7            : 1;
  __REG32  RMP8            : 1;
  __REG32  RMP9            : 1;
  __REG32  RMP10           : 1;
  __REG32  RMP11           : 1;
  __REG32  RMP12           : 1;
  __REG32  RMP13           : 1;
  __REG32  RMP14           : 1;
  __REG32  RMP15           : 1;
  __REG32  RMP16           : 1;
  __REG32  RMP17           : 1;
  __REG32  RMP18           : 1;
  __REG32  RMP19           : 1;
  __REG32  RMP20           : 1;
  __REG32  RMP21           : 1;
  __REG32  RMP22           : 1;
  __REG32  RMP23           : 1;
  __REG32  RMP24           : 1;
  __REG32  RMP25           : 1;
  __REG32  RMP26           : 1;
  __REG32  RMP27           : 1;
  __REG32  RMP28           : 1;
  __REG32  RMP29           : 1;
  __REG32  RMP30           : 1;
  __REG32  RMP31           : 1;
} __canrmp_bits;

/* CAN Receive Message Lost Register (RML) */
typedef struct {
  __REG32  RML0            : 1;
  __REG32  RML1            : 1;
  __REG32  RML2            : 1;
  __REG32  RML3            : 1;
  __REG32  RML4            : 1;
  __REG32  RML5            : 1;
  __REG32  RML6            : 1;
  __REG32  RML7            : 1;
  __REG32  RML8            : 1;
  __REG32  RML9            : 1;
  __REG32  RML10           : 1;
  __REG32  RML11           : 1;
  __REG32  RML12           : 1;
  __REG32  RML13           : 1;
  __REG32  RML14           : 1;
  __REG32  RML15           : 1;
  __REG32  RML16           : 1;
  __REG32  RML17           : 1;
  __REG32  RML18           : 1;
  __REG32  RML19           : 1;
  __REG32  RML20           : 1;
  __REG32  RML21           : 1;
  __REG32  RML22           : 1;
  __REG32  RML23           : 1;
  __REG32  RML24           : 1;
  __REG32  RML25           : 1;
  __REG32  RML26           : 1;
  __REG32  RML27           : 1;
  __REG32  RML28           : 1;
  __REG32  RML29           : 1;
  __REG32  RML30           : 1;
  __REG32  RML31           : 1;
} __canrml_bits;

/* CAN Remote Frame Pending Register (RFP) */
typedef struct {
  __REG32  RFP0            : 1;
  __REG32  RFP1            : 1;
  __REG32  RFP2            : 1;
  __REG32  RFP3            : 1;
  __REG32  RFP4            : 1;
  __REG32  RFP5            : 1;
  __REG32  RFP6            : 1;
  __REG32  RFP7            : 1;
  __REG32  RFP8            : 1;
  __REG32  RFP9            : 1;
  __REG32  RFP10           : 1;
  __REG32  RFP11           : 1;
  __REG32  RFP12           : 1;
  __REG32  RFP13           : 1;
  __REG32  RFP14           : 1;
  __REG32  RFP15           : 1;
  __REG32  RFP16           : 1;
  __REG32  RFP17           : 1;
  __REG32  RFP18           : 1;
  __REG32  RFP19           : 1;
  __REG32  RFP20           : 1;
  __REG32  RFP21           : 1;
  __REG32  RFP22           : 1;
  __REG32  RFP23           : 1;
  __REG32  RFP24           : 1;
  __REG32  RFP25           : 1;
  __REG32  RFP26           : 1;
  __REG32  RFP27           : 1;
  __REG32  RFP28           : 1;
  __REG32  RFP29           : 1;
  __REG32  RFP30           : 1;
  __REG32  RFP31           : 1;
} __canrfp_bits;

/* CAN Local Acceptance Mask (LAM) */
typedef struct {
  __REG32  LAM0            : 1;
  __REG32  LAM1            : 1;
  __REG32  LAM2            : 1;
  __REG32  LAM3            : 1;
  __REG32  LAM4            : 1;
  __REG32  LAM5            : 1;
  __REG32  LAM6            : 1;
  __REG32  LAM7            : 1;
  __REG32  LAM8            : 1;
  __REG32  LAM9            : 1;
  __REG32  LAM10           : 1;
  __REG32  LAM11           : 1;
  __REG32  LAM12           : 1;
  __REG32  LAM13           : 1;
  __REG32  LAM14           : 1;
  __REG32  LAM15           : 1;
  __REG32  LAM16           : 1;
  __REG32  LAM17           : 1;
  __REG32  LAM18           : 1;
  __REG32  LAM19           : 1;
  __REG32  LAM20           : 1;
  __REG32  LAM21           : 1;
  __REG32  LAM22           : 1;
  __REG32  LAM23           : 1;
  __REG32  LAM24           : 1;
  __REG32  LAM25           : 1;
  __REG32  LAM26           : 1;
  __REG32  LAM27           : 1;
  __REG32  LAM28           : 1;
  __REG32                  : 2;
  __REG32  LAMI            : 1;
} __canlam_bits;

/* CAN Global Acceptance Mask (GAM) */
typedef struct {
  __REG32  GAM0            : 1;
  __REG32  GAM1            : 1;
  __REG32  GAM2            : 1;
  __REG32  GAM3            : 1;
  __REG32  GAM4            : 1;
  __REG32  GAM5            : 1;
  __REG32  GAM6            : 1;
  __REG32  GAM7            : 1;
  __REG32  GAM8            : 1;
  __REG32  GAM9            : 1;
  __REG32  GAM10           : 1;
  __REG32  GAM11           : 1;
  __REG32  GAM12           : 1;
  __REG32  GAM13           : 1;
  __REG32  GAM14           : 1;
  __REG32  GAM15           : 1;
  __REG32  GAM16           : 1;
  __REG32  GAM17           : 1;
  __REG32  GAM18           : 1;
  __REG32  GAM19           : 1;
  __REG32  GAM20           : 1;
  __REG32  GAM21           : 1;
  __REG32  GAM22           : 1;
  __REG32  GAM23           : 1;
  __REG32  GAM24           : 1;
  __REG32  GAM25           : 1;
  __REG32  GAM26           : 1;
  __REG32  GAM27           : 1;
  __REG32  GAM28           : 1;
  __REG32                  : 2;
  __REG32  GAMI            : 1;
} __cangam_bits;

/* CAN Master Control Register (MCR) */
typedef struct {
  __REG32  SRES            : 1;
  __REG32  TSCC            : 1;
  __REG32                  : 1;
  __REG32  MTOS            : 1;
  __REG32  WUBA            : 1;
  __REG32                  : 1;
  __REG32  SMR             : 1;
  __REG32  CCR             : 1;
  __REG32  TSTERR          : 1;
  __REG32  TSTLB           : 1;
  __REG32                  : 1;
  __REG32  SUR             : 1;
  __REG32                  :20;
} __canmcr_bits;

/* CAN Bit Configuration Register 1 (BCR1) */
typedef struct {
  __REG32  BRP             : 8;
  __REG32                  :24;
} __canbcr1_bits;

/* CAN Bit Configuration Register 2 (BCR2) */
typedef struct {
  __REG32  TSEG1           : 4;
  __REG32  TSEG2           : 3;
  __REG32  SAM             : 1;
  __REG32  SJW             : 2;
  __REG32                  :22;
} __canbcr2_bits;

/* CAN Time Stamp Counter Register */
typedef struct {
  __REG32  TSC             :16;
  __REG32                  :16;
} __cantsc_bits;

/* CAN Time Stamp Counter Prescaler Register */
typedef struct {
  __REG32  TSP             : 4;
  __REG32                  :28;
} __cantsp_bits;

/* CAN Global Status Register (GSR) */
typedef struct {
  __REG32  EW              : 1;
  __REG32  EP              : 1;
  __REG32  BO              : 1;
  __REG32  TSO             : 1;
  __REG32                  : 2;
  __REG32  SMA             : 1;
  __REG32  CCE             : 1;
  __REG32  SUA             : 1;
  __REG32                  : 1;
  __REG32  TM              : 1;
  __REG32  RM              : 1;
  __REG32  MIS             : 5;
  __REG32                  :15;
} __cangsr_bits;

/* CAN Error Counter Register (CEC) */
typedef struct {
  __REG32  REC             : 8;
  __REG32  TEC             : 8;
  __REG32                  :16;
} __cancec_bits;

/* CAN Global Interrupt Flag Register (GIF) */
typedef struct {
  __REG32  WLIF            : 1;
  __REG32  EPIF            : 1;
  __REG32  BOIF            : 1;
  __REG32  TSOIF           : 1;
  __REG32  TRMABF          : 1;
  __REG32  RMLIF           : 1;
  __REG32  WUIF            : 1;
  __REG32  RFPF            : 1;
  __REG32                  :24;
} __cangif_bits;

/* CAN Global Interrupt Mask Register (GIM) */
typedef struct {
  __REG32  WLIM            : 1;
  __REG32  EPIM            : 1;
  __REG32  BOIM            : 1;
  __REG32  TSOIM           : 1;
  __REG32  TRMABM          : 1;
  __REG32  RMLIM           : 1;
  __REG32  WUIM            : 1;
  __REG32  RFPM            : 1;
  __REG32                  :24;
} __cangim_bits;

/* CAN Mailbox Interrupt Mask Register (MBIM) */
typedef struct {
  __REG32  MBIM0            : 1;
  __REG32  MBIM1            : 1;
  __REG32  MBIM2            : 1;
  __REG32  MBIM3            : 1;
  __REG32  MBIM4            : 1;
  __REG32  MBIM5            : 1;
  __REG32  MBIM6            : 1;
  __REG32  MBIM7            : 1;
  __REG32  MBIM8            : 1;
  __REG32  MBIM9            : 1;
  __REG32  MBIM10           : 1;
  __REG32  MBIM11           : 1;
  __REG32  MBIM12           : 1;
  __REG32  MBIM13           : 1;
  __REG32  MBIM14           : 1;
  __REG32  MBIM15           : 1;
  __REG32  MBIM16           : 1;
  __REG32  MBIM17           : 1;
  __REG32  MBIM18           : 1;
  __REG32  MBIM19           : 1;
  __REG32  MBIM20           : 1;
  __REG32  MBIM21           : 1;
  __REG32  MBIM22           : 1;
  __REG32  MBIM23           : 1;
  __REG32  MBIM24           : 1;
  __REG32  MBIM25           : 1;
  __REG32  MBIM26           : 1;
  __REG32  MBIM27           : 1;
  __REG32  MBIM28           : 1;
  __REG32  MBIM29           : 1;
  __REG32  MBIM30           : 1;
  __REG32  MBIM31           : 1;
} __canmbim_bits;

/* CAN Mailbox Interrupt Flag Registers (MBTIF) */
typedef struct {
  __REG32  MBTIF0            : 1;
  __REG32  MBTIF1            : 1;
  __REG32  MBTIF2            : 1;
  __REG32  MBTIF3            : 1;
  __REG32  MBTIF4            : 1;
  __REG32  MBTIF5            : 1;
  __REG32  MBTIF6            : 1;
  __REG32  MBTIF7            : 1;
  __REG32  MBTIF8            : 1;
  __REG32  MBTIF9            : 1;
  __REG32  MBTIF10           : 1;
  __REG32  MBTIF11           : 1;
  __REG32  MBTIF12           : 1;
  __REG32  MBTIF13           : 1;
  __REG32  MBTIF14           : 1;
  __REG32  MBTIF15           : 1;
  __REG32  MBTIF16           : 1;
  __REG32  MBTIF17           : 1;
  __REG32  MBTIF18           : 1;
  __REG32  MBTIF19           : 1;
  __REG32  MBTIF20           : 1;
  __REG32  MBTIF21           : 1;
  __REG32  MBTIF22           : 1;
  __REG32  MBTIF23           : 1;
  __REG32  MBTIF24           : 1;
  __REG32  MBTIF25           : 1;
  __REG32  MBTIF26           : 1;
  __REG32  MBTIF27           : 1;
  __REG32  MBTIF28           : 1;
  __REG32  MBTIF29           : 1;
  __REG32  MBTIF30           : 1;
  __REG32  			             : 1;
} __canmbtif_bits;

/* CAN Mailbox Interrupt Flag Registers (MBRIF) */
typedef struct {
  __REG32  MBRIF0            : 1;
  __REG32  MBRIF1            : 1;
  __REG32  MBRIF2            : 1;
  __REG32  MBRIF3            : 1;
  __REG32  MBRIF4            : 1;
  __REG32  MBRIF5            : 1;
  __REG32  MBRIF6            : 1;
  __REG32  MBRIF7            : 1;
  __REG32  MBRIF8            : 1;
  __REG32  MBRIF9            : 1;
  __REG32  MBRIF10           : 1;
  __REG32  MBRIF11           : 1;
  __REG32  MBRIF12           : 1;
  __REG32  MBRIF13           : 1;
  __REG32  MBRIF14           : 1;
  __REG32  MBRIF15           : 1;
  __REG32  MBRIF16           : 1;
  __REG32  MBRIF17           : 1;
  __REG32  MBRIF18           : 1;
  __REG32  MBRIF19           : 1;
  __REG32  MBRIF20           : 1;
  __REG32  MBRIF21           : 1;
  __REG32  MBRIF22           : 1;
  __REG32  MBRIF23           : 1;
  __REG32  MBRIF24           : 1;
  __REG32  MBRIF25           : 1;
  __REG32  MBRIF26           : 1;
  __REG32  MBRIF27           : 1;
  __REG32  MBRIF28           : 1;
  __REG32  MBRIF29           : 1;
  __REG32  MBRIF30           : 1;
  __REG32  MBRIF31           : 1;
} __canmbrif_bits;

/*SEMCR*/
typedef struct {
  __REG32  BCLR    	: 1;
  __REG32  SESTP  	: 1;
  __REG32  DLOOP   	: 1;
  __REG32           : 3;
  __REG32  OPMODE 	: 2;
  __REG32           :24;
} __semcr_bits;

/*SECR0*/
typedef struct {
  __REG32  SPOL    	: 1;
  __REG32  SPHA	  	: 1;
  __REG32  SBOS   	: 1;
  __REG32           : 1;
  __REG32  IFSPSE 	: 1;
  __REG32           : 3;
  __REG32  STFIE	 	: 1;
  __REG32           : 2;
  __REG32  SILIE	 	: 1;
  __REG32           :20;
} __secr0_bits;

/*SECR1*/
typedef struct {
  __REG32  SSZ     	: 5;
  __REG32           : 3;
  __REG32  SER		 	: 8;
  __REG32           :16;
} __secr1_bits;

/*SEFSR*/
typedef struct {
  __REG32  IFS     	:10;
  __REG32           :22;
} __sefsr_bits;

/*SSSR*/
typedef struct {
  __REG32  SSS     	: 8;
  __REG32           :24;
} __sssr_bits;

/*SESR*/
typedef struct {
  __REG32  SRRDY   	: 1;
  __REG32  STRDY   	: 1;
  __REG32  SIDLE   	: 1;
  __REG32  IFSD   	: 1;
  __REG32           :10;
  __REG32  RBSI   	: 1;
  __REG32  TBSI   	: 1;
  __REG32  SEILC   	: 1;
  __REG32  PAR    	: 1;
  __REG32  RBF    	: 1;
  __REG32  TBF    	: 1;
  __REG32           :12;
} __sesr_bits;

/*SEDR*/
typedef struct {
  __REG32  DR   	:16;
  __REG32           :16;
} __sedr_bits;

/*SERSR*/
typedef struct {
  __REG32  RS		   	:16;
  __REG32           :16;
} __sersr_bits;

/*SEFLR*/
typedef struct {
  __REG32  SRBFL  	: 5;
  __REG32           : 3;
  __REG32  STBFL  	: 5;
  __REG32           :19;
} __seflr_bits;

/*SEILR*/
typedef struct {
  __REG32  RXIFL  	: 5;
  __REG32           : 3;
  __REG32  TXIFL  	: 5;
  __REG32           :19;
} __seilr_bits;

/*SEPR*/
typedef struct {
  __REG32  SEISE  	: 1;
  __REG32  SEIE   	: 1;
  __REG32  SEEO   	: 1;
  __REG32           : 1;
  __REG32  SEEN   	: 1;
  __REG32  SEP01   	: 2;
  __REG32           :25;
} __sepr_bits;

/*SELCR*/
typedef struct {
  __REG32  SLB	  	: 1;
  __REG32  SLTB   	: 1;
  __REG32           :30;
} __selcr_bits;

/*SEDER*/
typedef struct {
  __REG32  SBD	  	:16;
  __REG32  SFL	  	: 5;
  __REG32  SCID	  	: 3;
  __REG32  SRBFL  	: 5;
  __REG32           : 3;
} __seder_bits;

/*SEEIC*/
typedef struct {
  __REG32  EIC	  	: 1;
  __REG32           :31;
} __seeic_bits;

/*SERIC*/
typedef struct {
  __REG32  RIC	  	: 1;
  __REG32           :31;
} __seric_bits;

/*SETIC*/
typedef struct {
  __REG32  TIC	  	: 1;
  __REG32           :31;
} __setic_bits;

/*RSLTn (A/D Conversion Result Register n, n = 0 to 13)*/
typedef struct {
  __REG32  OVWR     : 1;
  __REG32           :11;
  __REG32  PARITY   : 1;
  __REG32  PH       : 3;
  __REG32  AINS     : 4;
  __REG32  AD       :12;
} __adcrslt_bits;

/*SETI0 (A/D Conversion Channel Select Register 0)*/
typedef struct {
  __REG32  ADSI0    : 4;
  __REG32  ADSI1    : 4;
  __REG32  ADSI2    : 4;
  __REG32  ADSI3    : 4;
  __REG32  ADSI4    : 4;
  __REG32  ADSI5    : 4;
  __REG32  ADSI6    : 4;
  __REG32  ADSI7    : 4;
} __adcseti0_bits;

/*SETI1 (A/D Conversion Channel Select Register 1)*/
typedef struct {
  __REG32  ADSI8    : 4;
  __REG32  ADSI9    : 4;
  __REG32  ADSI10   : 4;
  __REG32  ADSI11   : 4;
  __REG32  ADSI12   : 4;
  __REG32  ADSI13   : 4;
  __REG32           : 8;
} __adcseti1_bits;

/*SETT (A/D Conversion PMD Trigger Select Register)*/
typedef struct {
  __REG32  ADST0    : 2;
  __REG32  ADST1    : 2;
  __REG32  ADST2    : 2;
  __REG32  ADST3    : 2;
  __REG32  ADST4    : 2;
  __REG32  ADST5    : 2;
  __REG32  ADST6    : 2;
  __REG32  ADST7    : 2;
  __REG32  ADST8    : 2;
  __REG32  ADST9    : 2;
  __REG32  ADST10   : 2;
  __REG32  ADST11   : 2;
  __REG32  ADST12   : 2;
  __REG32  ADSI13   : 2;
  __REG32           : 4;
} __adcsett_bits;

/*AD Conversion Mode Control Register 0 (ADMOD0)*/
typedef struct {
  __REG32  ADMD     : 5;
  __REG32  BIT12    : 1;
  __REG32           :26;
} __adcmod0_bits;

/*AD Conversion Mode Control Register 1 (ADMOD1)*/
typedef struct {
  __REG32  ADPE0    : 1;
  __REG32  ADPE1    : 1;
  __REG32  ADPE2    : 1;
  __REG32  ADPE3    : 1;
  __REG32  ADPE4    : 1;
  __REG32  ADPE5    : 1;
  __REG32  ADPE6    : 1;
  __REG32  ADPE7    : 1;
  __REG32  ADPE8    : 1;
  __REG32  ADPE9    : 1;
  __REG32  ADPE10   : 1;
  __REG32  ADPE11   : 1;
  __REG32  ADPE12   : 1;
  __REG32  ADPE13   : 1;
  __REG32           :18;
} __adcmod1_bits;

/*ENA (A/D Conversion Enable Register)*/
typedef struct {
  __REG32  ADEN     : 1;
  __REG32           :31;
} __adcena_bits;

/*FLG (A/D Conversion End Flag Register)*/
typedef struct {
  __REG32  ADF      : 1;
  __REG32           :31;
} __adcflg_bits;

/*WDT Counter Register (WDTCNT)*/
typedef struct {
  __REG32  CNT      :20;
  __REG32           :12;
} __wdtcnt_bits;

/*WDT Lower limit value Compare Register (WDTMIN)*/
typedef struct {
  __REG32  MIN      :20;
  __REG32           :12;
} __wdtmin_bits;

/*WDT Upper limit value Compare Register (WDTMAX)*/
typedef struct {
  __REG32  MAX      :20;
  __REG32           :12;
} __wdtmax_bits;

/*WDT Control Register (WDTCTL)*/
typedef struct {
  __REG32  WDTDIS   : 1;
  __REG32           :31;
} __wdtctl_bits;

/*WDT Command Register (WDTCMD)*/
typedef struct {
  __REG32  CMD      :16;
  __REG32           :16;
} __wdtcmd_bits;

/*FLMPSCR (Flash Memory Protect/Security Register)*/
typedef struct {
  __REG32  BP0      : 1;
  __REG32  BP1      : 1;
  __REG32  BP2      : 1;
  __REG32  BP3      : 1;
  __REG32  BP4      : 1;
  __REG32           : 3;
  __REG32  BP8      : 1;
  __REG32           :23;
} __flmpscr_bits;

/*FLMPD0 (Flash Memory Protect Disable Register 0)*/
typedef struct {
  __REG32  BPD      : 1;
  __REG32           :31;
} __flmpd0_bits;

/*FLMPD1 (Flash Memory Protect Disable Register 1)*/
typedef struct {
  __REG32  BP8      : 1;
  __REG32           :31;
} __flmpd1_bits;

/*FLMWCR1 (Flash Memory Write Control Register 1)*/
typedef struct {
  __REG32  HRST     : 3;
  __REG32           :29;
} __flmwcr1_bits;

/*FLMWCR3 (Flash Memory Write Control Register 3)*/
typedef struct {
  __REG32  RDBY     : 1;
  __REG32  CRB      : 1;
  __REG32  FRB      : 1;
  __REG32           :29;
} __flmwcr3_bits;

/*FLMCADR (Flash Memory Command Address Register)*/
typedef struct {
  __REG32  		      : 2;
  __REG32  FLMCADR  :19;
  __REG32           :11;
} __flmcadr_bits;

/*FLMWADR (Flash Memory Write Address Register)*/
typedef struct {
  __REG32  		      : 2;
  __REG32  FLMWADR  :19;
  __REG32           :11;
} __flmwadr_bits;

/*FLMWCAR (Flash Memory Write Cycle Adjustment Register FLASH)*/
typedef struct {
  __REG32  		      : 2;
  __REG32  FLMWADR  :19;
  __REG32           :11;
} __flmwcar_bits;

/*WDATCNT (Flash Memory Write Data Counter Register1 FLASH)*/
typedef struct {
  __REG32  WDATCNT  : 7;
  __REG32           :25;
} __wdatcnt_bits;

/*FLMWSR0 (Flash Memory Write Status Register 0)*/
typedef struct {
  __REG32  RDFA		  : 1;
  __REG32  TRMA		  : 1;
  __REG32  OPTA		  : 1;
  __REG32  UTRA		  : 1;
  __REG32           :28;
} __flmwsr0_bits;

/*FLMWSR1 (Flash Memory Write Status Register 1)*/
typedef struct {
  __REG32  RTRY		  : 1;
  __REG32           :31;
} __flmwsr1_bits;

/*FLMWSR3 (Flash Memory Write Status Register 3)*/
typedef struct {
  __REG32  BNK0 	  : 1;
  __REG32  BNK1		  : 1;
  __REG32           : 6;
  __REG32  BNK0CNT  : 8;
  __REG32           :16;
} __flmwsr3_bits;

/*FLMWSR4 (Flash Memory Write Status Register 4)*/
typedef struct {
  __REG32  FLMID	  :13;
  __REG32           :19;
} __flmwsr4_bits;

/*FLMWSR5 (Flash Memory Write Status Register 5)*/
typedef struct {
  __REG32  UFLG		  : 4;
  __REG32  ASWP0		: 2;
  __REG32  ASWP1		: 2;
  __REG32           :24;
} __flmwsr5_bits;

/*EXCCNT (EXCITER Control Register)*/
/*RCCCNTL (Rate Counter Compensation Control Register)*/
typedef struct {
  __REG32  EN			  : 1;
  __REG32           :31;
} __exccntl_bits;

/*CARCNTL (Carrier Control Register)*/
typedef struct {
  __REG32  RATE		  : 1;
  __REG32           :31;
} __carcntl_bits;

/*RATE (Rate Count Set Register)*/
/*CARSFTA (Carrier A Phase Amount Setting Register)*/
/*CARSFTB (Carrier B Phase Amount Setting Register)*/
/*CARSRC (Carrier Source Register)*/
/*CARREF (Reference Carrier Register)*/
/*RATE_N (Rate Count Set Register: Monitor Side Read)*/
/*CARSFTA_N (Carrier A Phase Amount Setting Register: Monitor Side Read)*/
/*CARSFTB_N (Carrier B Phase Amount Setting Register: Monitor Side Read)*/
/*CARSRC_N (Carrier Source Register: Monitor Side Read)*/
/*CARREF_N (Reference Carrier Register: Monitor Side Read)*/
typedef struct {
  __REG32  VAL		  :16;
  __REG32           :16;
} __rate_bits;

/*CARSET (Carrier Setting Register)*/
typedef struct {
  __REG32  REFBASE  : 4;
  __REG32           :28;
} __carset_bits;

/*INSDATn (Voltage Instruction Value Data Register n)*/
/*INSDATn_N (Voltage Instruction Value Data Register n: Monitor Side Read)*/
typedef struct {
  __REG32  VAL		  :15;
  __REG32           :17;
} __insdat_bits;

/*EXCCNTL_N (EXCITER Control Register: Monitor Side Read)*/
typedef struct {
  __REG32  EN			  : 1;
  __REG32           :31;
} __exccntl_n_bits;

/*CARCNTL_N (Carrier Control Register: Monitor Side Read)*/
typedef struct {
  __REG32  EN			  : 1;
  __REG32           :31;
} __carcntl_n_bits;

/*CARSET_N (Carrier Setting Register: Monitor Side Read)*/
typedef struct {
  __REG32  REFBASE 	: 4;
  __REG32           :28;
} __carset_n_bits;

/*CRCTYP (CRC Data Width Register)*/
typedef struct {
  __REG32  DT			 	: 2;
  __REG32           :30;
} __crctyp_bits;

/*CRCRST (CRC Calculation Result Storage Register)*/
typedef struct {
  __REG32  C		 		:16;
  __REG32           :16;
} __crcrst_bits;

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

/* Declarations common to compiler and assembler **************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(NVIC,                0xE000E004,__READ       ,__nvic_bits);
__IO_REG32_BIT(SYSTICKCSR,          0xE000E010,__READ_WRITE ,__systickcsr_bits);
__IO_REG32_BIT(SYSTICKRVR,          0xE000E014,__READ_WRITE ,__systickrvr_bits);
__IO_REG32_BIT(SYSTICKCVR,          0xE000E018,__READ_WRITE ,__systickcvr_bits);
__IO_REG32_BIT(SYSTICKCALVR,        0xE000E01C,__READ       ,__systickcalvr_bits);
__IO_REG32_BIT(SETENA0,             0xE000E100,__READ_WRITE ,__setena0_bits);
__IO_REG32_BIT(SETENA1,             0xE000E104,__READ_WRITE ,__setena1_bits);
__IO_REG32_BIT(CLRENA0,             0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(CLRENA1,             0xE000E184,__READ_WRITE ,__clrena1_bits);
__IO_REG32_BIT(SETPEND0,            0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(SETPEND1,            0xE000E204,__READ_WRITE ,__setpend1_bits);
__IO_REG32_BIT(CLRPEND0,            0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(CLRPEND1,            0xE000E284,__READ_WRITE ,__clrpend1_bits);
__IO_REG32_BIT(ACTIVE0,             0xE000E300,__READ       ,__active0_bits);
__IO_REG32_BIT(ACTIVE1,             0xE000E304,__READ       ,__active1_bits);
__IO_REG32_BIT(IP0,                 0xE000E400,__READ_WRITE ,__pri0_bits);
__IO_REG32_BIT(IP1,                 0xE000E404,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(IP2,                 0xE000E408,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(IP3,                 0xE000E40C,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(IP4,                 0xE000E410,__READ_WRITE ,__pri4_bits);
__IO_REG32_BIT(IP5,                 0xE000E414,__READ_WRITE ,__pri5_bits);
__IO_REG32_BIT(IP6,                 0xE000E418,__READ_WRITE ,__pri6_bits);
__IO_REG32_BIT(IP7,                 0xE000E41C,__READ_WRITE ,__pri7_bits);
__IO_REG32_BIT(IP8,                 0xE000E420,__READ_WRITE ,__pri8_bits);
__IO_REG32_BIT(IP9,                 0xE000E424,__READ_WRITE ,__pri9_bits);
__IO_REG32_BIT(IP10,                0xE000E428,__READ_WRITE ,__pri10_bits);
__IO_REG32_BIT(IP11,                0xE000E42C,__READ_WRITE ,__pri11_bits);
__IO_REG32_BIT(IP12,                0xE000E430,__READ_WRITE ,__pri12_bits);
__IO_REG32_BIT(IP13,                0xE000E434,__READ_WRITE ,__pri13_bits);
__IO_REG32_BIT(IP14,                0xE000E438,__READ_WRITE ,__pri14_bits);
__IO_REG32_BIT(IP15,                0xE000E43C,__READ_WRITE ,__pri15_bits);
__IO_REG32_BIT(CPUIDBR,             0xE000ED00,__READ       ,__cpuidbr_bits);
__IO_REG32_BIT(ICSR,                0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(VTOR,                0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,               0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,                 0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,                 0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR0,               0xE000ED18,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(SHPR1,               0xE000ED1C,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(SHPR2,               0xE000ED20,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(SHCSR,               0xE000ED24,__READ_WRITE ,__shcsr_bits);
__IO_REG32_BIT(CFSR,                0xE000ED28,__READ_WRITE ,__cfsr_bits);
__IO_REG32_BIT(HFSR,                0xE000ED2C,__READ_WRITE ,__hfsr_bits);
__IO_REG32_BIT(DFSR,                0xE000ED30,__READ_WRITE ,__dfsr_bits);
__IO_REG32(    MMFAR,               0xE000ED34,__READ_WRITE);
__IO_REG32(    BFAR,                0xE000ED38,__READ_WRITE);
__IO_REG32_BIT(STIR,                0xE000EF00,__WRITE      ,__stir_bits);

/***************************************************************************
 **
 ** CPU
 **
 ***************************************************************************/
__IO_REG32(    CPUID,               0xE00FE000,__READ       );

/***************************************************************************
 **
 ** EXCTL
 **
 ***************************************************************************/
__IO_REG32_BIT(TAS0,                0x4003F000,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS1,                0x4003F004,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS2,                0x4003F008,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS3,                0x4003F00C,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS4,                0x4003F010,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS5,                0x4003F014,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS6,                0x4003F018,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS7,                0x4003F01C,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS8,                0x4003F020,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS9,                0x4003F024,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS10,               0x4003F028,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS11,               0x4003F02C,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS12,               0x4003F030,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS13,               0x4003F034,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS14,               0x4003F038,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS15,               0x4003F03C,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS16,               0x4003F040,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS17,               0x4003F044,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS18,               0x4003F048,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS19,               0x4003F04C,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS20,               0x4003F050,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS21,               0x4003F054,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS22,               0x4003F058,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS23,               0x4003F05C,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS24,               0x4003F060,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS25,               0x4003F064,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS26,               0x4003F068,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS27,               0x4003F06C,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS28,               0x4003F070,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS29,               0x4003F074,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS30,               0x4003F078,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(TAS31,               0x4003F07C,__READ_WRITE ,__tas_bits);
__IO_REG32_BIT(EXINT0,              0x4003F080,__READ_WRITE ,__exint_bits);
__IO_REG32_BIT(EXINT1,              0x4003F084,__READ_WRITE ,__exint_bits);

/***************************************************************************
 **
 ** IOSET
 **
 ***************************************************************************/
__IO_REG32_BIT(FNCSELA,             0x40020000,__READ_WRITE ,__fncsela_bits);
__IO_REG32_BIT(FNCSELB,             0x40020004,__READ_WRITE ,__fncselb_bits);
__IO_REG32_BIT(FNCSELC,             0x40020008,__READ_WRITE ,__fncselc_bits);
__IO_REG32_BIT(FNCSELD,             0x4002000C,__READ_WRITE ,__fncseld_bits);
__IO_REG32_BIT(FNCSELE,             0x40020010,__READ_WRITE ,__fncsele_bits);
__IO_REG32_BIT(FNCSELF,             0x40020014,__READ_WRITE ,__fncself_bits);
__IO_REG32_BIT(FNCSELG,             0x40020018,__READ_WRITE ,__fncselg_bits);
__IO_REG32_BIT(FNCSELH,             0x4002001C,__READ_WRITE ,__fncselh_bits);
__IO_REG32_BIT(PDENA,             	0x40020100,__READ_WRITE ,__pdena_bits);
__IO_REG32_BIT(PDENB,             	0x40020104,__READ_WRITE ,__pdenb_bits);
__IO_REG32_BIT(PDEND,             	0x4002010C,__READ_WRITE ,__pdend_bits);
__IO_REG32_BIT(PDENE,             	0x40020110,__READ_WRITE ,__pdene_bits);
__IO_REG32_BIT(PDENF,             	0x40020114,__READ_WRITE ,__pdenf_bits);
__IO_REG32_BIT(PDENG,             	0x40020118,__READ_WRITE ,__pdeng_bits);
__IO_REG32_BIT(PDENH,             	0x4002011C,__READ_WRITE ,__pdenh_bits);
__IO_REG32_BIT(PUENA,             	0x40020180,__READ_WRITE ,__puena_bits);
__IO_REG32_BIT(PUENB,             	0x40020184,__READ_WRITE ,__puenb_bits);
__IO_REG32_BIT(PUEND,             	0x4002018C,__READ_WRITE ,__puend_bits);
__IO_REG32_BIT(PUENE,             	0x40020190,__READ_WRITE ,__puene_bits);
__IO_REG32_BIT(PUENF,             	0x40020194,__READ_WRITE ,__puenf_bits);
__IO_REG32_BIT(PUENG,             	0x40020198,__READ_WRITE ,__pueng_bits);
__IO_REG32_BIT(PUENH,             	0x4002019C,__READ_WRITE ,__puenh_bits);
__IO_REG32_BIT(SEICSCTL,            0x40020400,__READ_WRITE ,__seicsctl_bits);
__IO_REG32_BIT(CANM,             		0x40020500,__READ_WRITE ,__canm_bits);
__IO_REG32_BIT(SIOCKEN,             0x40020600,__READ_WRITE ,__siocken_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(MONA,           			0x40021000,__READ_WRITE ,__mona_bits);
__IO_REG32_BIT(OUTA,           			0x40021004,__READ_WRITE ,__outa_bits);
__IO_REG32_BIT(OENA,           			0x40021008,__READ_WRITE ,__oena_bits);
__IO_REG32_BIT(MONB,           			0x40021100,__READ_WRITE ,__monb_bits);
__IO_REG32_BIT(OUTB,           			0x40021104,__READ_WRITE ,__outb_bits);
__IO_REG32_BIT(OENB,           			0x40021108,__READ_WRITE ,__oenb_bits);
__IO_REG32_BIT(MONC,           			0x40021200,__READ_WRITE ,__monc_bits);
__IO_REG32_BIT(MOND,           			0x40021300,__READ_WRITE ,__mond_bits);
__IO_REG32_BIT(OUTD,           			0x40021304,__READ_WRITE ,__outd_bits);
__IO_REG32_BIT(OEND,           			0x40021308,__READ_WRITE ,__oend_bits);
__IO_REG32_BIT(MONE,           			0x40021400,__READ_WRITE ,__mone_bits);
__IO_REG32_BIT(OUTE,           			0x40021404,__READ_WRITE ,__oute_bits);
__IO_REG32_BIT(OENE,           			0x40021408,__READ_WRITE ,__oene_bits);
__IO_REG32_BIT(MONF,           			0x40021500,__READ_WRITE ,__monf_bits);
__IO_REG32_BIT(OUTF,           			0x40021504,__READ_WRITE ,__outf_bits);
__IO_REG32_BIT(OENF,           			0x40021508,__READ_WRITE ,__oenf_bits);
__IO_REG32_BIT(MONG,           			0x40021600,__READ_WRITE ,__mong_bits);
__IO_REG32_BIT(OUTG,           			0x40021604,__READ_WRITE ,__outg_bits);
__IO_REG32_BIT(OENG,           			0x40021608,__READ_WRITE ,__oeng_bits);
__IO_REG32_BIT(MONH,           			0x40021700,__READ_WRITE ,__monh_bits);
__IO_REG32_BIT(OUTH,           			0x40021704,__READ_WRITE ,__outh_bits);
__IO_REG32_BIT(OENH,           			0x40021708,__READ_WRITE ,__oenh_bits);
__IO_REG32_BIT(MONK,           			0x40021800,__READ_WRITE ,__monk_bits);

/***************************************************************************
 **
 ** TB00
 **
 ***************************************************************************/
__IO_REG32_BIT(TB00TRUN,           	0x40023000,__READ_WRITE ,__tbtrun_bits);
__IO_REG32_BIT(TB00TCR,            	0x40023004,__READ_WRITE ,__tbtcr_bits);
__IO_REG32(		 TB00TCNT,           	0x40023008,__READ_WRITE );
__IO_REG32_BIT(TB00CMP0CR,          0x40023800,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32_BIT(TB00CMP1CR,          0x40023804,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32(		 TB00CMP0,          	0x40023808,__READ_WRITE );
__IO_REG32(		 TB00CMP1,          	0x4002380C,__READ_WRITE );
__IO_REG32_BIT(TB00CMPDO0,          0x40023810,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB00CMPDO1,          0x40023814,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB00CMPSD0,          0x40023818,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB00CMPSD1,          0x4002381C,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB00CMPOM0,          0x40023820,__READ_WRITE ,__tbcmpom_bits);
__IO_REG32_BIT(TB00CMPOM1,          0x40023824,__READ_WRITE ,__tbcmpom_bits);

/***************************************************************************
 **
 ** TB01
 **
 ***************************************************************************/
__IO_REG32_BIT(TB01TRUN,           	0x40024000,__READ_WRITE ,__tbtrun_bits);
__IO_REG32_BIT(TB01TCR,            	0x40024004,__READ_WRITE ,__tbtcr_bits);
__IO_REG32(		 TB01TCNT,           	0x40024008,__READ_WRITE );
__IO_REG32_BIT(TB01CMP0CR,          0x40024800,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32_BIT(TB01CMP1CR,          0x40024804,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32(		 TB01CMP0,          	0x40024808,__READ_WRITE );
__IO_REG32(		 TB01CMP1,          	0x4002480C,__READ_WRITE );
__IO_REG32_BIT(TB01CMPDO0,          0x40024810,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB01CMPDO1,          0x40024814,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB01CMPSD0,          0x40024818,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB01CMPSD1,          0x4002481C,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB01CMPOM0,          0x40024820,__READ_WRITE ,__tbcmpom_bits);
__IO_REG32_BIT(TB01CMPOM1,          0x40024824,__READ_WRITE ,__tbcmpom_bits);

/***************************************************************************
 **
 ** TB02
 **
 ***************************************************************************/
__IO_REG32_BIT(TB02TRUN,           	0x40025000,__READ_WRITE ,__tbtrun_bits);
__IO_REG32_BIT(TB02TCR,            	0x40025004,__READ_WRITE ,__tbtcr_bits);
__IO_REG32(		 TB02TCNT,           	0x40025008,__READ_WRITE );
__IO_REG32_BIT(TB02CMP0CR,          0x40025800,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32_BIT(TB02CMP1CR,          0x40025804,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32(		 TB02CMP0,          	0x40025808,__READ_WRITE );
__IO_REG32(		 TB02CMP1,          	0x4002580C,__READ_WRITE );
__IO_REG32_BIT(TB02CMPDO0,          0x40025810,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB02CMPDO1,          0x40025814,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB02CMPSD0,          0x40025818,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB02CMPSD1,          0x4002581C,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB02CMPOM0,          0x40025820,__READ_WRITE ,__tbcmpom_bits);
__IO_REG32_BIT(TB02CMPOM1,          0x40025824,__READ_WRITE ,__tbcmpom_bits);

/***************************************************************************
 **
 ** TB03
 **
 ***************************************************************************/
__IO_REG32_BIT(TB03TRUN,           	0x40026000,__READ_WRITE ,__tbtrun_bits);
__IO_REG32_BIT(TB03TCR,            	0x40026004,__READ_WRITE ,__tbtcr_bits);
__IO_REG32(		 TB03TCNT,           	0x40026008,__READ_WRITE );
__IO_REG32_BIT(TB03CMP0CR,          0x40026800,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32_BIT(TB03CMP1CR,          0x40026804,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32(		 TB03CMP0,          	0x40026808,__READ_WRITE );
__IO_REG32(		 TB03CMP1,          	0x4002680C,__READ_WRITE );
__IO_REG32_BIT(TB03CMPDO0,          0x40026810,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB03CMPDO1,          0x40026814,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB03CMPSD0,          0x40026818,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB03CMPSD1,          0x4002681C,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB03CMPOM0,          0x40026820,__READ_WRITE ,__tbcmpom_bits);
__IO_REG32_BIT(TB03CMPOM1,          0x40026824,__READ_WRITE ,__tbcmpom_bits);

/***************************************************************************
 **
 ** TB04
 **
 ***************************************************************************/
__IO_REG32_BIT(TB04TRUN,           	0x40027000,__READ_WRITE ,__tbtrun_bits);
__IO_REG32_BIT(TB04TCR,            	0x40027004,__READ_WRITE ,__tbtcr_bits);
__IO_REG32(		 TB04TCNT,           	0x40027008,__READ_WRITE );
__IO_REG32_BIT(TB04CMP0CR,          0x40027800,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32_BIT(TB04CMP1CR,          0x40027804,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32(		 TB04CMP0,          	0x40027808,__READ_WRITE );
__IO_REG32(		 TB04CMP1,          	0x4002780C,__READ_WRITE );
__IO_REG32_BIT(TB04CMPDO0,          0x40027810,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB04CMPDO1,          0x40027814,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB04CMPSD0,          0x40027818,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB04CMPSD1,          0x4002781C,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB04CMPOM0,          0x40027820,__READ_WRITE ,__tbcmpom_bits);
__IO_REG32_BIT(TB04CMPOM1,          0x40027824,__READ_WRITE ,__tbcmpom_bits);

/***************************************************************************
 **
 ** TB05
 **
 ***************************************************************************/
__IO_REG32_BIT(TB05TRUN,           	0x40028000,__READ_WRITE ,__tbtrun_bits);
__IO_REG32_BIT(TB05TCR,            	0x40028004,__READ_WRITE ,__tbtcr_bits);
__IO_REG32(		 TB05TCNT,           	0x40028008,__READ_WRITE );
__IO_REG32_BIT(TB05CMP0CR,          0x40028800,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32_BIT(TB05CMP1CR,          0x40028804,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32(		 TB05CMP0,          	0x40028808,__READ_WRITE );
__IO_REG32(		 TB05CMP1,          	0x4002880C,__READ_WRITE );
__IO_REG32_BIT(TB05CMPDO0,          0x40028810,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB05CMPDO1,          0x40028814,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB05CMPSD0,          0x40028818,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB05CMPSD1,          0x4002881C,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB05CMPOM0,          0x40028820,__READ_WRITE ,__tbcmpom_bits);
__IO_REG32_BIT(TB05CMPOM1,          0x40028824,__READ_WRITE ,__tbcmpom_bits);

/***************************************************************************
 **
 ** TB06
 **
 ***************************************************************************/
__IO_REG32_BIT(TB06TRUN,           	0x40029000,__READ_WRITE ,__tbtrun_bits);
__IO_REG32_BIT(TB06TCR,            	0x40029004,__READ_WRITE ,__tbtcr_bits);
__IO_REG32(		 TB06TCNT,           	0x40029008,__READ_WRITE );
__IO_REG32_BIT(TB06CMP0CR,          0x40029800,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32_BIT(TB06CMP1CR,          0x40029804,__READ_WRITE ,__tbcmpcr_bits);
__IO_REG32(		 TB06CMP0,          	0x40029808,__READ_WRITE );
__IO_REG32(		 TB06CMP1,          	0x4002980C,__READ_WRITE );
__IO_REG32_BIT(TB06CMPDO0,          0x40029810,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB06CMPDO1,          0x40029814,__READ_WRITE ,__tbcmpdo_bits);
__IO_REG32_BIT(TB06CMPSD0,          0x40029818,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB06CMPSD1,          0x4002981C,__READ_WRITE ,__tbcmpsd_bits);
__IO_REG32_BIT(TB06CMPOM0,          0x40029820,__READ_WRITE ,__tbcmpom_bits);
__IO_REG32_BIT(TB06CMPOM1,          0x40029824,__READ_WRITE ,__tbcmpom_bits);

/***************************************************************************
 **
 ** DMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACEN,              0xE0042004,__READ_WRITE ,__dmacen_bits);
__IO_REG32_BIT(DMAREQ,              0xE0042008,__READ_WRITE ,__dmareq_bits);
__IO_REG32_BIT(DMASUS,              0xE004200C,__READ_WRITE ,__dmasus_bits);
__IO_REG32_BIT(DMAACT,              0xE0042010,__READ_WRITE ,__dmaact_bits);
__IO_REG32_BIT(DMAEND,              0xE0042014,__READ_WRITE ,__dmaend_bits);
__IO_REG32_BIT(DMAPRI,              0xE0042018,__READ_WRITE ,__dmapri_bits);
__IO_REG32_BIT(DMAENE,              0xE004201C,__READ_WRITE ,__dmaene_bits);
__IO_REG32(    DMADTAB,             0xE0042020,__READ_WRITE );
__IO_REG32(    DMAEVAD,             0xE0042024,__READ_WRITE );
__IO_REG32_BIT(DMACHN,              0xE0042028,__READ_WRITE ,__dmachn_bits);
__IO_REG32_BIT(DMAXFTYP,            0xE004202C,__READ       ,__dmaxftyp_bits);
__IO_REG32(    DMAXFSAD,            0xE0042030,__READ       );
__IO_REG32(    DMAXFDAD,            0xE0042034,__READ       );
__IO_REG32_BIT(DMAXFSIZ,            0xE0042038,__READ       ,__dmaxfsiz_bits);
__IO_REG32(    DMADSADS,            0xE004203C,__READ       );
__IO_REG32_BIT(DMADSNUM,            0xE0042040,__READ       ,__dmadsnum_bits);
__IO_REG32_BIT(DMALRQ,              0xE0042044,__READ_WRITE ,__dmalrq_bits);
__IO_REG32_BIT(DMAMSK,              0xE0042800,__READ_WRITE ,__dmamsk_bits);

/***************************************************************************
 **
 ** TMR
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR_TBTRUN,          0x4002C000,__READ_WRITE ,__tmr_tbtrun_bits);
__IO_REG32_BIT(TMR_TBTCR,           0x4002C004,__READ_WRITE ,__tmr_tbtcr_bits);
__IO_REG32(    TMR_TBTCNT,          0x4002C008,__READ_WRITE );
__IO_REG32_BIT(TMR_CAPCR,           0x4002C400,__READ_WRITE ,__tmr_capcr_bits);
__IO_REG32(    TMR_CAP0R,           0x4002C404,__READ_WRITE );
__IO_REG32(    TMR_CAP0F,           0x4002C408,__READ       );
__IO_REG32(    TMR_CAP1R,           0x4002C40C,__READ       );
__IO_REG32(    TMR_CAP1F,           0x4002C410,__READ       );
__IO_REG32(    TMR_CAP2R,           0x4002C414,__READ       );
__IO_REG32(    TMR_CAP2F,           0x4002C418,__READ       );
__IO_REG32(    TMR_CAP3R,           0x4002C41C,__READ       );
__IO_REG32(    TMR_CAP3F,           0x4002C420,__READ       );
__IO_REG32(    TMR_CAP4R,           0x4002C424,__READ       );
__IO_REG32(    TMR_CAP4F,           0x4002C428,__READ       );
__IO_REG32(    TMR_CAP5R,           0x4002C42C,__READ       );
__IO_REG32(    TMR_CAP5F,           0x4002C430,__READ       );
__IO_REG32(    TMR_CAP6R,           0x4002C434,__READ       );
__IO_REG32(    TMR_CAP6F,           0x4002C438,__READ       );
__IO_REG32(    TMR_CAP7R,           0x4002C43C,__READ       );
__IO_REG32(    TMR_CAP7F,           0x4002C440,__READ       );
__IO_REG32_BIT(TMR_CMP0CR,          0x4002C800,__READ_WRITE ,__tmr_cmpcr_bits);
__IO_REG32_BIT(TMR_CMP1CR,          0x4002C804,__READ_WRITE ,__tmr_cmpcr_bits);
__IO_REG32(    TMR_CMP0,            0x4002C808,__READ_WRITE );
__IO_REG32(    TMR_CMP1,            0x4002C80C,__READ_WRITE );

/***************************************************************************
 **
 ** PMD
 **
 ***************************************************************************/
__IO_REG32_BIT(PMD0MDEN,            0x4003D000,__READ_WRITE ,__pmdmden_bits);
__IO_REG32_BIT(PMD0PORTMD,          0x4003D004,__READ_WRITE ,__pmdportmd_bits);
__IO_REG32_BIT(PMD0MDCR,            0x4003D008,__READ_WRITE ,__pmdmdpr_bits);
__IO_REG32_BIT(PMD0CNTSTA,          0x4003D00C,__READ				,__pmdcntsta_bits);
__IO_REG32_BIT(PMD0MDCNT,           0x4003D010,__READ				,__pmdmdcnt_bits);
__IO_REG32_BIT(PMD0MDPRD,           0x4003D014,__READ_WRITE ,__pmdmdprd_bits);
__IO_REG32_BIT(PMD0CMPU,            0x4003D018,__READ_WRITE ,__pmdcmpu_bits);
__IO_REG32_BIT(PMD0CMPV,            0x4003D01C,__READ_WRITE ,__pmdcmpv_bits);
__IO_REG32_BIT(PMD0CMPW,            0x4003D020,__READ_WRITE ,__pmdcmpw_bits);
__IO_REG32_BIT(PMD0MODESEL,         0x4003D024,__READ_WRITE ,__pmdmodesel_bits);
__IO_REG32_BIT(PMD0MDOUT,           0x4003D028,__READ_WRITE ,__pmdmdout_bits);
__IO_REG32_BIT(PMD0MDPOT,           0x4003D02C,__READ_WRITE ,__pmdmdpot_bits);
__IO_REG32_BIT(PMD0EMGREL,          0x4003D030,__READ_WRITE ,__pmdemgrel_bits);
__IO_REG32_BIT(PMD0EMGCR,           0x4003D034,__READ_WRITE ,__pmdemgcr_bits);
__IO_REG32_BIT(PMD0EMGSTA,          0x4003D038,__READ				,__pmdemgsta_bits);
__IO_REG32_BIT(PMD0DTR,           	0x4003D044,__READ_WRITE ,__pmddtr_bits);
__IO_REG32_BIT(PMD0TRGCMP0,         0x4003D048,__READ_WRITE ,__pmdtrgcmp_bits);
__IO_REG32_BIT(PMD0TRGCMP1,         0x4003D04C,__READ_WRITE ,__pmdtrgcmp_bits);
__IO_REG32_BIT(PMD0TRGCMP2,         0x4003D050,__READ_WRITE ,__pmdtrgcmp_bits);
__IO_REG32_BIT(PMD0TRGCMP3,         0x4003D054,__READ_WRITE ,__pmdtrgcmp_bits);
__IO_REG32_BIT(PMD0TRGCR,         	0x4003D058,__READ_WRITE ,__pmdtrgcr_bits);
__IO_REG32_BIT(PMD0TRGMD,         	0x4003D05C,__READ_WRITE ,__pmdtrgmd_bits);
__IO_REG32_BIT(PMD0TRGSEL,          0x4003D060,__READ_WRITE ,__pmdtrgsel_bits);

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWMSYNCRUN,          0x4002B000,__READ_WRITE ,__pwmsyncrun_bits);

/***************************************************************************
 **
 ** PWM00
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM00_RUN,           0x4002A400,__READ_WRITE ,__pwm_run_bits);
__IO_REG32_BIT(PWM00_CTRL,          0x4002A404,__READ_WRITE ,__pwm_cr_bits);
__IO_REG32_BIT(PWM00_MOD,           0x4002A408,__READ_WRITE ,__pwm_mod_bits);
__IO_REG32_BIT(PWM00_OUTCTRL,       0x4002A40C,__READ_WRITE ,__pwm_outctrl_bits);
__IO_REG32_BIT(PWM00_PRICMP,        0x4002A410,__READ_WRITE ,__pwm_rg0_bits);
__IO_REG32_BIT(PWM00_DUTYCMP,       0x4002A414,__READ_WRITE ,__pwm_rg1_bits);
__IO_REG32_BIT(PWM00_CNT,           0x4002A418,__READ_WRITE ,__pwm_cnt_bits);

/***************************************************************************
 **
 ** PWM01
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM01_RUN,           0x4002A800,__READ_WRITE ,__pwm_run_bits);
__IO_REG32_BIT(PWM01_CTRL,          0x4002A804,__READ_WRITE ,__pwm_cr_bits);
__IO_REG32_BIT(PWM01_MOD,           0x4002A808,__READ_WRITE ,__pwm_mod_bits);
__IO_REG32_BIT(PWM01_OUTCTRL,       0x4002A80C,__READ_WRITE ,__pwm_outctrl_bits);
__IO_REG32_BIT(PWM01_PRICMP,        0x4002A810,__READ_WRITE ,__pwm_rg0_bits);
__IO_REG32_BIT(PWM01_DUTYCMP,       0x4002A814,__READ_WRITE ,__pwm_rg1_bits);
__IO_REG32_BIT(PWM01_CNT,           0x4002A818,__READ_WRITE ,__pwm_cnt_bits);

/***************************************************************************
 **
 ** PWM02
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM02_RUN,           0x4002AC00,__READ_WRITE ,__pwm_run_bits);
__IO_REG32_BIT(PWM02_CTRL,          0x4002AC04,__READ_WRITE ,__pwm_cr_bits);
__IO_REG32_BIT(PWM02_MOD,           0x4002AC08,__READ_WRITE ,__pwm_mod_bits);
__IO_REG32_BIT(PWM02_OUTCTRL,       0x4002AC0C,__READ_WRITE ,__pwm_outctrl_bits);
__IO_REG32_BIT(PWM02_PRICMP,        0x4002AC10,__READ_WRITE ,__pwm_rg0_bits);
__IO_REG32_BIT(PWM02_DUTYCMP,       0x4002AC14,__READ_WRITE ,__pwm_rg1_bits);
__IO_REG32_BIT(PWM02_CNT,           0x4002AC18,__READ_WRITE ,__pwm_cnt_bits);

/***************************************************************************
 **
 ** PWM10
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM10_RUN,           0x4002B400,__READ_WRITE ,__pwm_run_bits);
__IO_REG32_BIT(PWM10_CTRL,          0x4002B404,__READ_WRITE ,__pwm_cr_bits);
__IO_REG32_BIT(PWM10_MOD,           0x4002B408,__READ_WRITE ,__pwm_mod_bits);
__IO_REG32_BIT(PWM10_OUTCTRL,       0x4002B40C,__READ_WRITE ,__pwm_outctrl_bits);
__IO_REG32_BIT(PWM10_PRICMP,        0x4002B410,__READ_WRITE ,__pwm_rg0_bits);
__IO_REG32_BIT(PWM10_DUTYCMP,       0x4002B414,__READ_WRITE ,__pwm_rg1_bits);
__IO_REG32_BIT(PWM10_CNT,           0x4002B418,__READ_WRITE ,__pwm_cnt_bits);

/***************************************************************************
 **
 ** PWM11
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM11RUN,           0x4002B800,__READ_WRITE ,__pwm_run_bits);
__IO_REG32_BIT(PWM11CTRL,          0x4002B804,__READ_WRITE ,__pwm_cr_bits);
__IO_REG32_BIT(PWM11MOD,           0x4002B808,__READ_WRITE ,__pwm_mod_bits);
__IO_REG32_BIT(PWM11OUTCTRL,       0x4002B80C,__READ_WRITE ,__pwm_outctrl_bits);
__IO_REG32_BIT(PWM11PRICMP,        0x4002B810,__READ_WRITE ,__pwm_rg0_bits);
__IO_REG32_BIT(PWM11DUTYCMP,       0x4002B814,__READ_WRITE ,__pwm_rg1_bits);
__IO_REG32_BIT(PWM11CNT,           0x4002B818,__READ_WRITE ,__pwm_cnt_bits);

/***************************************************************************
 **
 ** PWM12
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM12RUN,           0x4002BC00,__READ_WRITE ,__pwm_run_bits);
__IO_REG32_BIT(PWM12CTRL,          0x4002BC04,__READ_WRITE ,__pwm_cr_bits);
__IO_REG32_BIT(PWM12MOD,           0x4002BC08,__READ_WRITE ,__pwm_mod_bits);
__IO_REG32_BIT(PWM12OUTCTRL,       0x4002BC0C,__READ_WRITE ,__pwm_outctrl_bits);
__IO_REG32_BIT(PWM12PRICMP,        0x4002BC10,__READ_WRITE ,__pwm_rg0_bits);
__IO_REG32_BIT(PWM12DUTYCMP,       0x4002BC14,__READ_WRITE ,__pwm_rg1_bits);
__IO_REG32_BIT(PWM12CNT,           0x4002BC18,__READ_WRITE ,__pwm_cnt_bits);

/***************************************************************************
 **
 ** SIO0
 **
 ***************************************************************************/
__IO_REG8_BIT( SC0EN,             	0x4003C000,__READ_WRITE ,__scen_bits);
__IO_REG8(		 SC0BUF,             	0x4003C004,__READ_WRITE );
__IO_REG8_BIT( SC0CR,             	0x4003C008,__READ_WRITE ,__sccr_bits);
__IO_REG8_BIT( SC0MOD0,            	0x4003C00C,__READ_WRITE ,__scmod0_bits);
__IO_REG8_BIT( SC0BRCR,           	0x4003C010,__READ_WRITE ,__scbrcr_bits);
__IO_REG8_BIT( SC0BRADD,           	0x4003C014,__READ_WRITE ,__scbradd_bits);
__IO_REG8_BIT( SC0MOD1,           	0x4003C018,__READ_WRITE ,__scmod1_bits);
__IO_REG8_BIT( SC0MOD2,           	0x4003C01C,__READ_WRITE ,__scmod2_bits);
__IO_REG8_BIT( SC0RFC,             	0x4003C020,__READ_WRITE ,__scrfc_bits);
__IO_REG8_BIT( SC0TFC,             	0x4003C024,__READ_WRITE ,__sctfc_bits);
__IO_REG8_BIT( SC0RST,             	0x4003C028,__READ				,__scrst_bits);
__IO_REG8_BIT( SC0TST,             	0x4003C02C,__READ				,__sctst_bits);
__IO_REG8_BIT( SC0FCNF,            	0x4003C030,__READ_WRITE ,__scfcnf_bits);

/***************************************************************************
 **
 ** SIO1
 **
 ***************************************************************************/
__IO_REG8_BIT( SC1EN,             	0x4003E000,__READ_WRITE ,__scen_bits);
__IO_REG8(		 SC1BUF,             	0x4003E004,__READ_WRITE );
__IO_REG8_BIT( SC1CR,             	0x4003E008,__READ_WRITE ,__sccr_bits);
__IO_REG8_BIT( SC1MOD0,            	0x4003E00C,__READ_WRITE ,__scmod0_bits);
__IO_REG8_BIT( SC1BRCR,           	0x4003E010,__READ_WRITE ,__scbrcr_bits);
__IO_REG8_BIT( SC1BRADD,           	0x4003E014,__READ_WRITE ,__scbradd_bits);
__IO_REG8_BIT( SC1MOD1,           	0x4003E018,__READ_WRITE ,__scmod1_bits);
__IO_REG8_BIT( SC1MOD2,           	0x4003E01C,__READ_WRITE ,__scmod2_bits);
__IO_REG8_BIT( SC1RFC,             	0x4003E020,__READ_WRITE ,__scrfc_bits);
__IO_REG8_BIT( SC1TFC,             	0x4003E024,__READ_WRITE ,__sctfc_bits);
__IO_REG8_BIT( SC1RST,             	0x4003E028,__READ				,__scrst_bits);
__IO_REG8_BIT( SC1TST,             	0x4003E02C,__READ				,__sctst_bits);
__IO_REG8_BIT( SC1FCNF,            	0x4003E030,__READ_WRITE ,__scfcnf_bits);

/***************************************************************************
 **
 ** SSP0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP0CR0,             0x40032000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP0CR1,             0x40032004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP0DR,              0x40032008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP0SR,              0x4003200C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP0CPSR,            0x40032010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP0IMSC,            0x40032014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP0RIS,             0x40032018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP0MIS,             0x4003201C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP0ICR,             0x40032020,__WRITE      ,__sspicr_bits);
__IO_REG32_BIT(SSP0DMACR,           0x40032024,__READ_WRITE ,__sspdmacr_bits);
__IO_REG32(    SSP0RXINTCLR,        0x40032028,__WRITE      );
__IO_REG32(    SSP0TXINTCLR,        0x4003202C,__WRITE      );
__IO_REG32(    SSP0RORINTCLR,       0x40032030,__WRITE      );
__IO_REG32(    SSP0RTINTCLR,        0x40032034,__WRITE      );

/***************************************************************************
 **
 ** SSP1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP1CR0,             0x40033000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP1CR1,             0x40033004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP1DR,              0x40033008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP1SR,              0x4003300C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP1CPSR,            0x40033010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP1IMSC,            0x40033014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP1RIS,             0x40033018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP1MIS,             0x4003301C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP1ICR,             0x40033020,__WRITE      ,__sspicr_bits);
__IO_REG32_BIT(SSP1DMACR,           0x40033024,__READ_WRITE ,__sspdmacr_bits);
__IO_REG32(    SSP1RXINTCLR,        0x40033028,__WRITE      );
__IO_REG32(    SSP1TXINTCLR,        0x4003302C,__WRITE      );
__IO_REG32(    SSP1RORINTCLR,       0x40033030,__WRITE      );
__IO_REG32(    SSP1RTINTCLR,        0x40033034,__WRITE      );

/***************************************************************************
 **
 ** SSP2
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP2CR0,             0x40034000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP2CR1,             0x40034004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP2DR,              0x40034008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP2SR,              0x4003400C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP2CPSR,            0x40034010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP2IMSC,            0x40034014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP2RIS,             0x40034018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP2MIS,             0x4003401C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP2ICR,             0x40034020,__WRITE      ,__sspicr_bits);
__IO_REG32_BIT(SSP2DMACR,           0x40034024,__READ_WRITE ,__sspdmacr_bits);
__IO_REG32(    SSP2RXINTCLR,        0x40034028,__WRITE      );
__IO_REG32(    SSP2TXINTCLR,        0x4003402C,__WRITE      );
__IO_REG32(    SSP2RORINTCLR,       0x40034030,__WRITE      );
__IO_REG32(    SSP2RTINTCLR,        0x40034034,__WRITE      );

/***************************************************************************
 **
 ** SSP3
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP3CR0,             0x40035000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP3CR1,             0x40035004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP3DR,              0x40035008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP3SR,              0x4003500C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP3CPSR,            0x40035010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP3IMSC,            0x40035014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP3RIS,             0x40035018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP3MIS,             0x4003501C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP3ICR,             0x40035020,__WRITE      ,__sspicr_bits);
__IO_REG32_BIT(SSP3DMACR,           0x40035024,__READ_WRITE ,__sspdmacr_bits);
__IO_REG32(    SSP3RXINTCLR,        0x40035028,__WRITE      );
__IO_REG32(    SSP3TXINTCLR,        0x4003502C,__WRITE      );
__IO_REG32(    SSP3RORINTCLR,       0x40035030,__WRITE      );
__IO_REG32(    SSP3RTINTCLR,        0x40035034,__WRITE      );

/***************************************************************************
 **
 ** SSP4
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP4CR0,             0x40036000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP4CR1,             0x40036004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP4DR,              0x40036008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP4SR,              0x4003600C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP4CPSR,            0x40036010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP4IMSC,            0x40036014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP4RIS,             0x40036018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP4MIS,             0x4003601C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP4ICR,             0x40036020,__WRITE      ,__sspicr_bits);
__IO_REG32_BIT(SSP4DMACR,           0x40036024,__READ_WRITE ,__sspdmacr_bits);
__IO_REG32(    SSP4RXINTCLR,        0x40036028,__WRITE      );
__IO_REG32(    SSP4TXINTCLR,        0x4003602C,__WRITE      );
__IO_REG32(    SSP4RORINTCLR,       0x40036030,__WRITE      );
__IO_REG32(    SSP4RTINTCLR,        0x40036034,__WRITE      );

/***************************************************************************
 **
 ** CAN0
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN0MB0ID,           0x400C0000,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB0CR,           0x400C0008,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB0DL,           0x400C0010,__READ_WRITE );
__IO_REG32(    CAN0MB0DH,           0x400C0018,__READ_WRITE );
__IO_REG32_BIT(CAN0MB1ID,           0x400C0020,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB1CR,           0x400C0028,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB1DL,           0x400C0030,__READ_WRITE );
__IO_REG32(    CAN0MB1DH,           0x400C0038,__READ_WRITE );
__IO_REG32_BIT(CAN0MB2ID,           0x400C0040,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB2CR,           0x400C0048,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB2DL,           0x400C0050,__READ_WRITE );
__IO_REG32(    CAN0MB2DH,           0x400C0058,__READ_WRITE );
__IO_REG32_BIT(CAN0MB3ID,           0x400C0060,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB3CR,           0x400C0068,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB3DL,           0x400C0070,__READ_WRITE );
__IO_REG32(    CAN0MB3DH,           0x400C0078,__READ_WRITE );
__IO_REG32_BIT(CAN0MB4ID,           0x400C0080,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB4CR,           0x400C0088,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB4DL,           0x400C0090,__READ_WRITE );
__IO_REG32(    CAN0MB4DH,           0x400C0098,__READ_WRITE );
__IO_REG32_BIT(CAN0MB5ID,           0x400C00A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB5CR,           0x400C00A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB5DL,           0x400C00B0,__READ_WRITE );
__IO_REG32(    CAN0MB5DH,           0x400C00B8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB6ID,           0x400C00C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB6CR,           0x400C00C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB6DL,           0x400C00D0,__READ_WRITE );
__IO_REG32(    CAN0MB6DH,           0x400C00D8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB7ID,           0x400C00E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB7CR,           0x400C00E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB7DL,           0x400C00F0,__READ_WRITE );
__IO_REG32(    CAN0MB7DH,           0x400C00F8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB8ID,           0x400C0100,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB8CR,           0x400C0108,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB8DL,           0x400C0110,__READ_WRITE );
__IO_REG32(    CAN0MB8DH,           0x400C0118,__READ_WRITE );
__IO_REG32_BIT(CAN0MB9ID,           0x400C0120,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB9CR,           0x400C0128,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB9DL,           0x400C0130,__READ_WRITE );
__IO_REG32(    CAN0MB9DH,           0x400C0138,__READ_WRITE );
__IO_REG32_BIT(CAN0MB10ID,          0x400C0140,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB10CR,          0x400C0148,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB10DL,          0x400C0150,__READ_WRITE );
__IO_REG32(    CAN0MB10DH,          0x400C0158,__READ_WRITE );
__IO_REG32_BIT(CAN0MB11ID,          0x400C0160,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB11CR,          0x400C0168,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB11DL,          0x400C0170,__READ_WRITE );
__IO_REG32(    CAN0MB11DH,          0x400C0178,__READ_WRITE );
__IO_REG32_BIT(CAN0MB12ID,          0x400C0180,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB12CR,          0x400C0188,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB12DL,          0x400C0190,__READ_WRITE );
__IO_REG32(    CAN0MB12DH,          0x400C0198,__READ_WRITE );
__IO_REG32_BIT(CAN0MB13ID,          0x400C01A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB13CR,          0x400C01A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB13DL,          0x400C01B0,__READ_WRITE );
__IO_REG32(    CAN0MB13DH,          0x400C01B8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB14ID,          0x400C01C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB14CR,          0x400C01C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB14DL,          0x400C01D0,__READ_WRITE );
__IO_REG32(    CAN0MB14DH,          0x400C01D8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB15ID,          0x400C01E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB15CR,          0x400C01E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB15DL,          0x400C01F0,__READ_WRITE );
__IO_REG32(    CAN0MB15DH,          0x400C01F8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB16ID,          0x400C0200,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB16CR,          0x400C0208,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB16DL,          0x400C0210,__READ_WRITE );
__IO_REG32(    CAN0MB16DH,          0x400C0218,__READ_WRITE );
__IO_REG32_BIT(CAN0MB17ID,          0x400C0220,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB17CR,          0x400C0228,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB17DL,          0x400C0230,__READ_WRITE );
__IO_REG32(    CAN0MB17DH,          0x400C0238,__READ_WRITE );
__IO_REG32_BIT(CAN0MB18ID,          0x400C0240,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB18CR,          0x400C0248,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB18DL,          0x400C0250,__READ_WRITE );
__IO_REG32(    CAN0MB18DH,          0x400C0258,__READ_WRITE );
__IO_REG32_BIT(CAN0MB19ID,          0x400C0260,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB19CR,          0x400C0268,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB19DL,          0x400C0270,__READ_WRITE );
__IO_REG32(    CAN0MB19DH,          0x400C0278,__READ_WRITE );
__IO_REG32_BIT(CAN0MB20ID,          0x400C0280,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB20CR,          0x400C0288,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB20DL,          0x400C0290,__READ_WRITE );
__IO_REG32(    CAN0MB20DH,          0x400C0298,__READ_WRITE );
__IO_REG32_BIT(CAN0MB21ID,          0x400C02A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB21CR,          0x400C02A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB21DL,          0x400C02B0,__READ_WRITE );
__IO_REG32(    CAN0MB21DH,          0x400C02B8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB22ID,          0x400C02C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB22CR,          0x400C02C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB22DL,          0x400C02D0,__READ_WRITE );
__IO_REG32(    CAN0MB22DH,          0x400C02D8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB23ID,          0x400C02E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB23CR,          0x400C02E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB23DL,          0x400C02F0,__READ_WRITE );
__IO_REG32(    CAN0MB23DH,          0x400C02F8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB24ID,          0x400C0300,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB24CR,          0x400C0308,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB24DL,          0x400C0310,__READ_WRITE );
__IO_REG32(    CAN0MB24DH,          0x400C0318,__READ_WRITE );
__IO_REG32_BIT(CAN0MB25ID,          0x400C0320,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB25CR,          0x400C0328,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB25DL,          0x400C0330,__READ_WRITE );
__IO_REG32(    CAN0MB25DH,          0x400C0338,__READ_WRITE );
__IO_REG32_BIT(CAN0MB26ID,          0x400C0340,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB26CR,          0x400C0348,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB26DL,          0x400C0350,__READ_WRITE );
__IO_REG32(    CAN0MB26DH,          0x400C0358,__READ_WRITE );
__IO_REG32_BIT(CAN0MB27ID,          0x400C0360,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB27CR,          0x400C0368,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB27DL,          0x400C0370,__READ_WRITE );
__IO_REG32(    CAN0MB27DH,          0x400C0378,__READ_WRITE );
__IO_REG32_BIT(CAN0MB28ID,          0x400C0380,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB28CR,          0x400C0388,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB28DL,          0x400C0390,__READ_WRITE );
__IO_REG32(    CAN0MB28DH,          0x400C0398,__READ_WRITE );
__IO_REG32_BIT(CAN0MB29ID,          0x400C03A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB29CR,          0x400C03A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB29DL,          0x400C03B0,__READ_WRITE );
__IO_REG32(    CAN0MB29DH,          0x400C03B8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB30ID,          0x400C03C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB30CR,          0x400C03C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB30DL,          0x400C03D0,__READ_WRITE );
__IO_REG32(    CAN0MB30DH,          0x400C03D8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB31ID,          0x400C03E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB31CR,          0x400C03E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB31DL,          0x400C03F0,__READ_WRITE );
__IO_REG32(    CAN0MB31DH,          0x400C03F8,__READ_WRITE );
__IO_REG32_BIT(CAN0MC,              0x400C0400,__READ_WRITE ,__canmc_bits);
__IO_REG32_BIT(CAN0MD,              0x400C0404,__READ_WRITE ,__canmd_bits);
__IO_REG32_BIT(CAN0TRS,             0x400C0410,__READ_WRITE ,__cantrs_bits);
__IO_REG32_BIT(CAN0TRR,             0x400C0418,__READ_WRITE ,__cantrr_bits);
__IO_REG32_BIT(CAN0TA,              0x400C0420,__READ_WRITE ,__canta_bits);
__IO_REG32_BIT(CAN0AA,              0x400C0428,__READ_WRITE ,__canaa_bits);
__IO_REG32_BIT(CAN0RMP,             0x400C0430,__READ_WRITE ,__canrmp_bits);
__IO_REG32_BIT(CAN0RML,             0x400C0438,__READ_WRITE ,__canrml_bits);
__IO_REG32_BIT(CAN0LAM,             0x400C0440,__READ_WRITE ,__canlam_bits);
__IO_REG32_BIT(CAN0GAM,             0x400C0448,__READ_WRITE ,__cangam_bits);
__IO_REG32_BIT(CAN0MCR,             0x400C0450,__READ_WRITE ,__canmcr_bits);
__IO_REG32_BIT(CAN0GSR,             0x400C0458,__READ       ,__cangsr_bits);
__IO_REG32_BIT(CAN0BCR1,            0x400C0460,__READ_WRITE ,__canbcr1_bits);
__IO_REG32_BIT(CAN0BCR2,            0x400C0468,__READ_WRITE ,__canbcr2_bits);
__IO_REG32_BIT(CAN0GIF,             0x400C0470,__READ_WRITE ,__cangif_bits);
__IO_REG32_BIT(CAN0GIM,             0x400C0478,__READ_WRITE ,__cangim_bits);
__IO_REG32_BIT(CAN0MBTIF,           0x400C0480,__READ_WRITE ,__canmbtif_bits);
__IO_REG32_BIT(CAN0MBRIF,           0x400C0488,__READ_WRITE ,__canmbrif_bits);
__IO_REG32_BIT(CAN0MBIM,            0x400C0490,__READ_WRITE ,__canmbim_bits);
__IO_REG32_BIT(CAN0CDR,             0x400C0498,__READ_WRITE ,__cancdr_bits);
__IO_REG32_BIT(CAN0RFP,             0x400C04A0,__READ_WRITE ,__canrfp_bits);
__IO_REG32_BIT(CAN0CEC,             0x400C04A8,__READ       ,__cancec_bits);
__IO_REG32_BIT(CAN0TSP,             0x400C04B0,__READ_WRITE ,__cantsp_bits);
__IO_REG32_BIT(CAN0TSC,             0x400C04B8,__READ_WRITE ,__cantsc_bits);
__IO_REG32(		 CAN0INTCRCLR,        0x400C0800,__READ_WRITE );
__IO_REG32(		 CAN0INTCTCLR,        0x400C0808,__READ_WRITE );
__IO_REG32(		 CAN0INTCGCLR,        0x400C0810,__READ_WRITE );

/***************************************************************************
 **
 ** CAN1
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN1MB0ID,           0x400C1000,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB0CR,           0x400C1008,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB0DL,           0x400C1010,__READ_WRITE );
__IO_REG32(    CAN1MB0DH,           0x400C1018,__READ_WRITE );
__IO_REG32_BIT(CAN1MB1ID,           0x400C1020,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB1CR,           0x400C1028,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB1DL,           0x400C1030,__READ_WRITE );
__IO_REG32(    CAN1MB1DH,           0x400C1038,__READ_WRITE );
__IO_REG32_BIT(CAN1MB2ID,           0x400C1040,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB2CR,           0x400C1048,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB2DL,           0x400C1050,__READ_WRITE );
__IO_REG32(    CAN1MB2DH,           0x400C1058,__READ_WRITE );
__IO_REG32_BIT(CAN1MB3ID,           0x400C1060,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB3CR,           0x400C1068,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB3DL,           0x400C1070,__READ_WRITE );
__IO_REG32(    CAN1MB3DH,           0x400C1078,__READ_WRITE );
__IO_REG32_BIT(CAN1MB4ID,           0x400C1080,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB4CR,           0x400C1088,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB4DL,           0x400C1090,__READ_WRITE );
__IO_REG32(    CAN1MB4DH,           0x400C1098,__READ_WRITE );
__IO_REG32_BIT(CAN1MB5ID,           0x400C10A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB5CR,           0x400C10A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB5DL,           0x400C10B0,__READ_WRITE );
__IO_REG32(    CAN1MB5DH,           0x400C10B8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB6ID,           0x400C10C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB6CR,           0x400C10C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB6DL,           0x400C10D0,__READ_WRITE );
__IO_REG32(    CAN1MB6DH,           0x400C10D8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB7ID,           0x400C10E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB7CR,           0x400C10E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB7DL,           0x400C10F0,__READ_WRITE );
__IO_REG32(    CAN1MB7DH,           0x400C10F8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB8ID,           0x400C1100,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB8CR,           0x400C1108,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB8DL,           0x400C1110,__READ_WRITE );
__IO_REG32(    CAN1MB8DH,           0x400C1118,__READ_WRITE );
__IO_REG32_BIT(CAN1MB9ID,           0x400C1120,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB9CR,           0x400C1128,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB9DL,           0x400C1130,__READ_WRITE );
__IO_REG32(    CAN1MB9DH,           0x400C1138,__READ_WRITE );
__IO_REG32_BIT(CAN1MB10ID,          0x400C1140,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB10CR,          0x400C1148,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB10DL,          0x400C1150,__READ_WRITE );
__IO_REG32(    CAN1MB10DH,          0x400C1158,__READ_WRITE );
__IO_REG32_BIT(CAN1MB11ID,          0x400C1160,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB11CR,          0x400C1168,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB11DL,          0x400C1170,__READ_WRITE );
__IO_REG32(    CAN1MB11DH,          0x400C1178,__READ_WRITE );
__IO_REG32_BIT(CAN1MB12ID,          0x400C1180,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB12CR,          0x400C1188,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB12DL,          0x400C1190,__READ_WRITE );
__IO_REG32(    CAN1MB12DH,          0x400C1198,__READ_WRITE );
__IO_REG32_BIT(CAN1MB13ID,          0x400C11A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB13CR,          0x400C11A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB13DL,          0x400C11B0,__READ_WRITE );
__IO_REG32(    CAN1MB13DH,          0x400C11B8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB14ID,          0x400C11C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB14CR,          0x400C11C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB14DL,          0x400C11D0,__READ_WRITE );
__IO_REG32(    CAN1MB14DH,          0x400C11D8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB15ID,          0x400C11E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB15CR,          0x400C11E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB15DL,          0x400C11F0,__READ_WRITE );
__IO_REG32(    CAN1MB15DH,          0x400C11F8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB16ID,          0x400C1200,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB16CR,          0x400C1208,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB16DL,          0x400C1210,__READ_WRITE );
__IO_REG32(    CAN1MB16DH,          0x400C1218,__READ_WRITE );
__IO_REG32_BIT(CAN1MB17ID,          0x400C1220,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB17CR,          0x400C1228,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB17DL,          0x400C1230,__READ_WRITE );
__IO_REG32(    CAN1MB17DH,          0x400C1238,__READ_WRITE );
__IO_REG32_BIT(CAN1MB18ID,          0x400C1240,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB18CR,          0x400C1248,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB18DL,          0x400C1250,__READ_WRITE );
__IO_REG32(    CAN1MB18DH,          0x400C1258,__READ_WRITE );
__IO_REG32_BIT(CAN1MB19ID,          0x400C1260,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB19CR,          0x400C1268,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB19DL,          0x400C1270,__READ_WRITE );
__IO_REG32(    CAN1MB19DH,          0x400C1278,__READ_WRITE );
__IO_REG32_BIT(CAN1MB20ID,          0x400C1280,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB20CR,          0x400C1288,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB20DL,          0x400C1290,__READ_WRITE );
__IO_REG32(    CAN1MB20DH,          0x400C1298,__READ_WRITE );
__IO_REG32_BIT(CAN1MB21ID,          0x400C12A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB21CR,          0x400C12A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB21DL,          0x400C12B0,__READ_WRITE );
__IO_REG32(    CAN1MB21DH,          0x400C12B8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB22ID,          0x400C12C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB22CR,          0x400C12C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB22DL,          0x400C12D0,__READ_WRITE );
__IO_REG32(    CAN1MB22DH,          0x400C12D8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB23ID,          0x400C12E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB23CR,          0x400C12E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB23DL,          0x400C12F0,__READ_WRITE );
__IO_REG32(    CAN1MB23DH,          0x400C12F8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB24ID,          0x400C1300,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB24CR,          0x400C1308,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB24DL,          0x400C1310,__READ_WRITE );
__IO_REG32(    CAN1MB24DH,          0x400C1318,__READ_WRITE );
__IO_REG32_BIT(CAN1MB25ID,          0x400C1320,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB25CR,          0x400C1328,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB25DL,          0x400C1330,__READ_WRITE );
__IO_REG32(    CAN1MB25DH,          0x400C1338,__READ_WRITE );
__IO_REG32_BIT(CAN1MB26ID,          0x400C1340,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB26CR,          0x400C1348,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB26DL,          0x400C1350,__READ_WRITE );
__IO_REG32(    CAN1MB26DH,          0x400C1358,__READ_WRITE );
__IO_REG32_BIT(CAN1MB27ID,          0x400C1360,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB27CR,          0x400C1368,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB27DL,          0x400C1370,__READ_WRITE );
__IO_REG32(    CAN1MB27DH,          0x400C1378,__READ_WRITE );
__IO_REG32_BIT(CAN1MB28ID,          0x400C1380,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB28CR,          0x400C1388,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB28DL,          0x400C1390,__READ_WRITE );
__IO_REG32(    CAN1MB28DH,          0x400C1398,__READ_WRITE );
__IO_REG32_BIT(CAN1MB29ID,          0x400C13A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB29CR,          0x400C13A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB29DL,          0x400C13B0,__READ_WRITE );
__IO_REG32(    CAN1MB29DH,          0x400C13B8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB30ID,          0x400C13C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB30CR,          0x400C13C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB30DL,          0x400C13D0,__READ_WRITE );
__IO_REG32(    CAN1MB30DH,          0x400C13D8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB31ID,          0x400C13E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB31CR,          0x400C13E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB31DL,          0x400C13F0,__READ_WRITE );
__IO_REG32(    CAN1MB31DH,          0x400C13F8,__READ_WRITE );
__IO_REG32_BIT(CAN1MC,              0x400C1400,__READ_WRITE ,__canmc_bits);
__IO_REG32_BIT(CAN1MD,              0x400C1404,__READ_WRITE ,__canmd_bits);
__IO_REG32_BIT(CAN1TRS,             0x400C1410,__READ_WRITE ,__cantrs_bits);
__IO_REG32_BIT(CAN1TRR,             0x400C1418,__READ_WRITE ,__cantrr_bits);
__IO_REG32_BIT(CAN1TA,              0x400C1420,__READ_WRITE ,__canta_bits);
__IO_REG32_BIT(CAN1AA,              0x400C1428,__READ_WRITE ,__canaa_bits);
__IO_REG32_BIT(CAN1RMP,             0x400C1430,__READ_WRITE ,__canrmp_bits);
__IO_REG32_BIT(CAN1RML,             0x400C1438,__READ_WRITE ,__canrml_bits);
__IO_REG32_BIT(CAN1LAM,             0x400C1440,__READ_WRITE ,__canlam_bits);
__IO_REG32_BIT(CAN1GAM,             0x400C1448,__READ_WRITE ,__cangam_bits);
__IO_REG32_BIT(CAN1MCR,             0x400C1450,__READ_WRITE ,__canmcr_bits);
__IO_REG32_BIT(CAN1GSR,             0x400C1458,__READ       ,__cangsr_bits);
__IO_REG32_BIT(CAN1BCR1,            0x400C1460,__READ_WRITE ,__canbcr1_bits);
__IO_REG32_BIT(CAN1BCR2,            0x400C1468,__READ_WRITE ,__canbcr2_bits);
__IO_REG32_BIT(CAN1GIF,             0x400C1470,__READ_WRITE ,__cangif_bits);
__IO_REG32_BIT(CAN1GIM,             0x400C1478,__READ_WRITE ,__cangim_bits);
__IO_REG32_BIT(CAN1MBTIF,           0x400C1480,__READ_WRITE ,__canmbtif_bits);
__IO_REG32_BIT(CAN1MBRIF,           0x400C1488,__READ_WRITE ,__canmbrif_bits);
__IO_REG32_BIT(CAN1MBIM,            0x400C1490,__READ_WRITE ,__canmbim_bits);
__IO_REG32_BIT(CAN1CDR,             0x400C1498,__READ_WRITE ,__cancdr_bits);
__IO_REG32_BIT(CAN1RFP,             0x400C14A0,__READ_WRITE ,__canrfp_bits);
__IO_REG32_BIT(CAN1CEC,             0x400C14A8,__READ       ,__cancec_bits);
__IO_REG32_BIT(CAN1TSP,             0x400C14B0,__READ_WRITE ,__cantsp_bits);
__IO_REG32_BIT(CAN1TSC,             0x400C14B8,__READ_WRITE ,__cantsc_bits);
__IO_REG32(		 CAN1INTCRCLR,        0x400C1800,__READ_WRITE );
__IO_REG32(		 CAN1INTCTCLR,        0x400C1808,__READ_WRITE );
__IO_REG32(		 CAN1INTCGCLR,        0x400C1810,__READ_WRITE );

/***************************************************************************
 **
 ** ESEI
 **
 ***************************************************************************/
__IO_REG32_BIT(SEMCR,           		0x400C2000,__READ_WRITE ,__semcr_bits);
__IO_REG32_BIT(SECR0,           		0x400C2004,__READ_WRITE ,__secr0_bits);
__IO_REG32_BIT(SECR1,           		0x400C2008,__READ_WRITE ,__secr1_bits);
__IO_REG32_BIT(SEFSR,           		0x400C200C,__READ_WRITE ,__sefsr_bits);
__IO_REG32_BIT(SSSR,           			0x400C2010,__READ_WRITE ,__sssr_bits);
__IO_REG32_BIT(SESR,           			0x400C2014,__READ_WRITE ,__sesr_bits);
__IO_REG32_BIT(SEDR,           			0x400C2018,__READ_WRITE ,__sedr_bits);
__IO_REG32_BIT(SERSR,           		0x400C201C,__READ				,__sersr_bits);
__IO_REG32_BIT(SEFLR,           		0x400C2020,__READ				,__seflr_bits);
__IO_REG32_BIT(SEILR,           		0x400C2024,__READ_WRITE ,__seilr_bits);
__IO_REG32_BIT(SEPR,           			0x400C2028,__READ_WRITE ,__sepr_bits);
__IO_REG32_BIT(SELCR,           		0x400C202C,__READ_WRITE ,__selcr_bits);
__IO_REG32_BIT(SEDER,           		0x400C2030,__READ_WRITE ,__seder_bits);
__IO_REG32_BIT(SEEIC,           		0x400C2800,__WRITE 			,__seeic_bits);
__IO_REG32_BIT(SERIC,           		0x400C2804,__WRITE 			,__seric_bits);
__IO_REG32_BIT(SETIC,           		0x400C2808,__WRITE 			,__setic_bits);

/***************************************************************************
 **
 ** ADC0
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC0RSLT0,           0x4002F000,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT1,           0x4002F004,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT2,           0x4002F008,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT3,           0x4002F00C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT4,           0x4002F010,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT5,           0x4002F014,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT6,           0x4002F018,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT7,           0x4002F01C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT8,           0x4002F020,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT9,           0x4002F024,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT10,          0x4002F028,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT11,          0x4002F02C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT12,          0x4002F030,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT13,          0x4002F034,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0SETI0,          	0x4002F038,__READ_WRITE ,__adcseti0_bits);
__IO_REG32_BIT(ADC0SETI1,         	0x4002F03C,__READ_WRITE ,__adcseti1_bits);
__IO_REG32_BIT(ADC0SETT,         		0x4002F040,__READ_WRITE ,__adcsett_bits);
__IO_REG32_BIT(ADC0MOD0,            0x4002F044,__READ_WRITE ,__adcmod0_bits);
__IO_REG32_BIT(ADC0MOD1,            0x4002F048,__READ_WRITE ,__adcmod1_bits);
__IO_REG32_BIT(ADC0ENA,             0x4002F04C,__READ_WRITE ,__adcena_bits);
__IO_REG32_BIT(ADC0FLG,             0x4002F050,__READ       ,__adcflg_bits);

/***************************************************************************
 **
 ** ADC1
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC1RSLT0,           0x4002D000,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT1,           0x4002D004,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT2,           0x4002D008,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT3,           0x4002D00C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT4,           0x4002D010,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT5,           0x4002D014,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT6,           0x4002D018,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT7,           0x4002D01C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT8,           0x4002D020,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT9,           0x4002D024,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT10,          0x4002D028,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT11,          0x4002D02C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT12,          0x4002D030,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT13,          0x4002D034,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1SETI0,          	0x4002D038,__READ_WRITE ,__adcseti0_bits);
__IO_REG32_BIT(ADC1SETI1,         	0x4002D03C,__READ_WRITE ,__adcseti1_bits);
__IO_REG32_BIT(ADC1SETT,         		0x4002D040,__READ_WRITE ,__adcsett_bits);
__IO_REG32_BIT(ADC1MOD0,            0x4002D044,__READ_WRITE ,__adcmod0_bits);
__IO_REG32_BIT(ADC1MOD1,            0x4002D048,__READ_WRITE ,__adcmod1_bits);
__IO_REG32_BIT(ADC1ENA,             0x4002D04C,__READ_WRITE ,__adcena_bits);
__IO_REG32_BIT(ADC1FLG,             0x4002D050,__READ       ,__adcflg_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WDTCNT,              0x4002E000,__READ       ,__wdtcnt_bits);
__IO_REG32_BIT(WDTMIN,              0x4002E004,__READ_WRITE ,__wdtmin_bits);
__IO_REG32_BIT(WDTMAX,              0x4002E008,__READ_WRITE ,__wdtmax_bits);
__IO_REG32_BIT(WDTCTL,              0x4002E00C,__READ_WRITE ,__wdtctl_bits);
__IO_REG32_BIT(WDTCMD,              0x4002E010,__WRITE      ,__wdtcmd_bits);

/***************************************************************************
 **
 ** FLM
 **
 ***************************************************************************/
__IO_REG32_BIT(FLMPSCR,             0x400C4000,__READ_WRITE ,__flmpscr_bits);
__IO_REG32_BIT(FLMPD0,          	  0x400C4004,__READ_WRITE ,__flmpd0_bits);
__IO_REG32_BIT(FLMPD1,             	0x400C4008,__READ_WRITE ,__flmpd1_bits);
__IO_REG32_BIT(FLMWCR1,             0x400C4010,__READ_WRITE ,__flmwcr1_bits);
__IO_REG32_BIT(FLMWCR3,             0x400C4018,__READ				,__flmwcr3_bits);
__IO_REG32_BIT(FLMCADR,             0x400C401C,__READ_WRITE ,__flmcadr_bits);
__IO_REG32(		 FLMCDTR,             0x400C4020,__READ_WRITE );
__IO_REG32_BIT(FLMWADR,             0x400C4024,__READ_WRITE ,__flmwadr_bits);
__IO_REG32(		 FLMWDTR,             0x400C4028,__READ_WRITE );
__IO_REG32_BIT(FLMWCAR,             0x400C402C,__READ_WRITE ,__flmwcar_bits);
__IO_REG32_BIT(WDATCNT,             0x400C4030,__READ_WRITE ,__wdatcnt_bits);
__IO_REG32_BIT(FLMWSR0,             0x400C4034,__READ				,__flmwsr0_bits);
__IO_REG32_BIT(FLMWSR1,             0x400C4038,__READ_WRITE ,__flmwsr1_bits);
__IO_REG32_BIT(FLMWSR3,             0x400C4040,__READ_WRITE ,__flmwsr3_bits);
__IO_REG32_BIT(FLMWSR4,             0x400C4044,__READ_WRITE ,__flmwsr4_bits);
__IO_REG32_BIT(FLMWSR5,             0x400C4048,__READ_WRITE ,__flmwsr5_bits);

/***************************************************************************
 **
 ** EXCITER
 **
 ***************************************************************************/
__IO_REG32_BIT(EXCCNTL,            	0x40022000,__READ_WRITE ,__exccntl_bits);
__IO_REG32_BIT(RCCCNTL,            	0x40022004,__READ_WRITE ,__exccntl_bits);
__IO_REG32_BIT(CARCNTL,            	0x40022008,__READ_WRITE ,__carcntl_bits);
__IO_REG32_BIT(RATE,            		0x4002200C,__READ_WRITE ,__rate_bits);
__IO_REG32(		 RCOUNT,            	0x40022010,__READ_WRITE );
__IO_REG32_BIT(CARSET,            	0x40022014,__READ_WRITE ,__carset_bits);
__IO_REG32_BIT(CARSFTA,            	0x40022018,__READ_WRITE ,__rate_bits);
__IO_REG32_BIT(CARSFTB,            	0x4002201C,__READ_WRITE ,__rate_bits);
__IO_REG32_BIT(CARSRC,            	0x40022020,__READ_WRITE ,__rate_bits);
__IO_REG32_BIT(CARREF,            	0x40022024,__READ				,__rate_bits);
__IO_REG32_BIT(INSDAT0,            	0x40022028,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT1,            	0x4002202C,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT2,            	0x40022030,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT3,            	0x40022034,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT4,            	0x40022038,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT5,            	0x4002203C,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT6,            	0x40022040,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT7,            	0x40022044,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(EXCCNTL_N,           0x40022800,__READ				,__exccntl_n_bits);
//__IO_REG32_BIT(RCCCNTL_N,           0x40022804,__READ_WRITE ,__rcccntl_n_bits);
__IO_REG32_BIT(CARCNTL_N,           0x40022808,__READ				,__carcntl_n_bits);
__IO_REG32_BIT(RATE_N,           		0x4002280C,__READ				,__rate_bits);
__IO_REG32(		 RCOUNT_N,           	0x40022810,__READ				);
__IO_REG32_BIT(CARSET_N,           	0x40022814,__READ				,__carset_n_bits);
__IO_REG32_BIT(CARSFTA_N,           0x40022818,__READ				,__rate_bits);
__IO_REG32_BIT(CARSFTB_N,           0x4002281C,__READ				,__rate_bits);
__IO_REG32_BIT(CARSRC_N,           	0x40022820,__READ				,__rate_bits);
__IO_REG32_BIT(CARREF_N,           	0x40022824,__READ       ,__insdat_bits);
__IO_REG32_BIT(INSDAT0_N,          	0x40022828,__READ       ,__insdat_bits);
__IO_REG32_BIT(INSDAT1_N,          	0x4002282C,__READ       ,__insdat_bits);
__IO_REG32_BIT(INSDAT2_N,          	0x40022830,__READ       ,__insdat_bits);
__IO_REG32_BIT(INSDAT3_N,          	0x40022834,__READ       ,__insdat_bits);
__IO_REG32_BIT(INSDAT4_N,          	0x40022838,__READ       ,__insdat_bits);
__IO_REG32_BIT(INSDAT5_N,          	0x4002283C,__READ       ,__insdat_bits);
__IO_REG32_BIT(INSDAT6_N,          	0x40022840,__READ       ,__insdat_bits);
__IO_REG32_BIT(INSDAT7_N,          	0x40022844,__READ       ,__insdat_bits);

/***************************************************************************
 **
 ** CRC
 **
 ***************************************************************************/
__IO_REG32(		 CRCDIN,             	0x400C4100,__READ_WRITE );
__IO_REG32_BIT(CRCTYP,          	  0x400C4114,__READ_WRITE ,__crctyp_bits);
__IO_REG32_BIT(CRCRST,             	0x400C4128,__READ_WRITE ,__crcrst_bits);

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  TMPM350FDTFG DMA Lines
 **
 ***************************************************************************/
#define DMA_ADC0    			0
#define DMA_ADC1    			1
#define DMA_TIMER00_M0		2
#define DMA_TIMER00_M1		3
#define DMA_TIMER01_M0		4
#define DMA_TIMER01_M1		5
#define DMA_TIMER02_M0		6
#define DMA_TIMER02_M1		7
#define DMA_TIMER03_M0		8
#define DMA_TIMER03_M1		9
#define DMA_PMD_M0				10
#define DMA_PMD_M1				11
#define DMA_PWM0					12
#define DMA_PWM1					13
#define DMA_PWM2					14
#define DMA_PWM3					15
#define DMA_PWM4					16
#define DMA_PWM5					17
#define DMA_TIMER07_RC0		18
#define DMA_TIMER07_FC0		19
#define DMA_TIMER07_RC1		20
#define DMA_TIMER07_FC1		21
#define DMA_TIMER07_RC2		22
#define DMA_TIMER07_FC2		23
#define DMA_TIMER07_RC3		24
#define DMA_TIMER07_FC3		25
#define DMA_UART0_RX			26
#define DMA_UART0_TX			27
#define DMA_UART1_RX			28
#define DMA_UART1_TX			29
#define DMA_SEI_RX				30
#define DMA_SEI_TX				31

/***************************************************************************
 **
 **  TMPM350FDTFG Interrupt Lines
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
#define INTERR                16
#define INTINFO               18
#define INTWDTERR             19
#define INTEMG                20
#define INTPMD                21
#define INTOVF0               24
#define INTCMP00              25
#define INTCMP01              26
#define INTOVF1               27
#define INTCMP10              28
#define INTCMP11              29
#define INTOVF2               30
#define INTCMP20              31
#define INTCMP21              32
#define INTOVF3               33
#define INTCMP30              34
#define INTCMP31              35
#define INTOVF4               36
#define INTCMP40              37
#define INTCMP41              38
#define INTOVF5               39
#define INTCMP50              40
#define INTCMP51              41
#define INTOVF6               42
#define INTCMP60              43
#define INTCMP61              44
#define INTTBTOVF             45
#define INTTBTI0              46
#define INTTBTI1              47
#define INTTBTI2              48
#define INTTBTI3              49
#define INTTCAP0R             50
#define INTTCAP0F             51
#define INTTCAP1R             52
#define INTTCAP1F             53
#define INTTCAP2R             54
#define INTTCAP2F             55
#define INTTCAP3R             56
#define INTTCAP3F             57
#define INTTCAP4R             58
#define INTTCAP4F             59
#define INTTCAP5R             60
#define INTTCAP5F             61
#define INTTCAP6R             62
#define INTTCAP6F             63
#define INTTCAP7R             64
#define INTTCAP7F             65
#define INTTCMP0              66
#define INTTCMP1              67
#define INTADC0               68
#define INTADC1               69
#define INTDMAXFEND           70
#define INTRX0                71
#define INTTX0                72
#define INTRX1                73
#define INTTX1                74
#define INTESEIERR            75
#define INTESEIRX             76
#define INTESEITX             77
#define INTCAN0CR             78
#define INTCAN0CT             79
#define INTCAN0CG             80
#define INTCAN1CR             81
#define INTCAN1CT             82
#define INTCAN1CG             83
#define INTPWM00P             84
#define INTPWM01P             85
#define INTPWM02P             86
#define INTPWM10P             87
#define INTPWM11P             88
#define INTPWM12P             89
#define INTEXC                90
#endif    /* __IOTMPM350FDTFG_H */

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
Interrupt9   = INTERR         0x40
Interrupt10  = INTINFO        0x48
Interrupt11  = INTWDTERR      0x4C
Interrupt12  = INTEMG         0x50
Interrupt13  = INTPMD         0x54
Interrupt14  = INTOVF0        0x60
Interrupt15  = INTCMP00       0x64
Interrupt16  = INTCMP01       0x68
Interrupt17  = INTOVF1        0x6C
Interrupt18  = INTCMP10       0x70
Interrupt19  = INTCMP11       0x74
Interrupt20  = INTOVF2        0x78
Interrupt21  = INTCMP20       0x7C
Interrupt22  = INTCMP21       0x80
Interrupt23  = INTOVF3        0x84
Interrupt24  = INTCMP30       0x88
Interrupt25  = INTCMP31       0x8C
Interrupt26  = INTOVF4        0x90
Interrupt27  = INTCMP40       0x94
Interrupt28  = INTCMP41       0x98
Interrupt29  = INTOVF5        0x9C
Interrupt30  = INTCMP50       0xA0
Interrupt31  = INTCMP51       0xA4
Interrupt32  = INTOVF6        0xA8
Interrupt33  = INTCMP60       0xAC
Interrupt34  = INTCMP61       0xB0
Interrupt35  = INTTBTOVF      0xB4
Interrupt36  = INTTBTI0       0xB8
Interrupt37  = INTTBTI1       0xBC
Interrupt38  = INTTBTI2       0xC0
Interrupt39  = INTTBTI3       0xC4
Interrupt40  = INTTCAP0R      0xC8
Interrupt41  = INTTCAP0F      0xCC
Interrupt42  = INTTCAP1R      0xD0
Interrupt43  = INTTCAP1F      0xD4
Interrupt44  = INTTCAP2R      0xD8
Interrupt45  = INTTCAP2F      0xDC
Interrupt46  = INTTCAP3R      0xE0
Interrupt47  = INTTCAP3F      0xE4
Interrupt48  = INTTCAP4R      0xE8
Interrupt49  = INTTCAP4F      0xEC
Interrupt50  = INTTCAP5R      0xF0
Interrupt51  = INTTCAP5F      0xF4
Interrupt52  = INTTCAP6R      0xF8
Interrupt53  = INTTCAP6F      0xFC
Interrupt54  = INTTCAP7R      0x100
Interrupt55  = INTTCAP7F      0x104
Interrupt56  = INTTCMP0       0x108
Interrupt57  = INTTCMP1       0x10C
Interrupt58  = INTADC0        0x110
Interrupt59  = INTADC1        0x114
Interrupt60  = INTDMAXFEND    0x118
Interrupt61  = INTRX0         0x11C
Interrupt62  = INTTX0         0x120
Interrupt63  = INTRX1         0x124
Interrupt64  = INTTX1         0x128
Interrupt65  = INTESEIERR     0x12C
Interrupt66  = INTESEIRX      0x130
Interrupt67  = INTESEITX      0x134
Interrupt68  = INTCAN0CR      0x138
Interrupt69  = INTCAN0CT      0x13C
Interrupt70  = INTCAN0CG      0x140
Interrupt71  = INTCAN1CR      0x144
Interrupt72  = INTCAN1CT      0x148
Interrupt73  = INTCAN1CG      0x14C
Interrupt74  = INTPWM00P      0x150
Interrupt75  = INTPWM01P      0x154
Interrupt76  = INTPWM02P      0x158
Interrupt77  = INTPWM10P      0x15C
Interrupt78  = INTPWM11P      0x160
Interrupt79  = INTPWM12P      0x164
Interrupt80  = INTEXC         0x168
###DDF-INTERRUPT-END###*/