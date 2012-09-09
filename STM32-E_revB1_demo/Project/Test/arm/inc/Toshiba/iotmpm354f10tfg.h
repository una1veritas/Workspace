/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Toshiba TMPM354F10TFG
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 46813 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOTMPM354F10TFG_H
#define __IOTMPM354F10TFG_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMPM354F10TFG SPECIAL FUNCTION REGISTERS
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

/*FNCSELA*/
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
  __REG32 PD7  : 1;
  __REG32      :24;
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
  __REG32 PU7  : 1;
  __REG32      :24;
} __puena_bits;

/*FNCSELB*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32 FS3  : 2;
  __REG32 FS4  : 2;
  __REG32      :22;
} __fncselb_bits;

/*PDENB*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32 PD3  : 1;
  __REG32 PD4  : 1;
  __REG32      :27;
} __pdenb_bits;

/*PUENB*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32 PU3  : 1;
  __REG32 PU4  : 1;
  __REG32      :27;
} __puenb_bits;

/*FNCSELC*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32 FS3  : 2;
  __REG32 FS4  : 2;
  __REG32 FS5  : 2;
  __REG32 FS6  : 2;
  __REG32      :18;
} __fncselc_bits;

/*PDENC*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32 PD3  : 1;
  __REG32 PD4  : 1;
  __REG32 PD5  : 1;
  __REG32 PD6  : 1;
  __REG32      :25;
} __pdenc_bits;

/*PUENC*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32 PU3  : 1;
  __REG32 PU4  : 1;
  __REG32 PU5  : 1;
  __REG32 PU6  : 1;
  __REG32      :25;
} __puenc_bits;

/*FNCSELD*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32 FS3  : 2;
  __REG32 FS4  : 2;
  __REG32 FS5  : 2;
  __REG32      :20;
} __fncseld_bits;

/*PDEND*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32 PD3  : 1;
  __REG32 PD4  : 1;
  __REG32 PD5  : 1;
  __REG32      :26;
} __pdend_bits;

/*PUEND*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32 PU3  : 1;
  __REG32 PU4  : 1;
  __REG32 PU5  : 1;
  __REG32      :26;
} __puend_bits;

/*FNCSELE*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32 FS3  : 2;
  __REG32 FS4  : 2;
  __REG32      :22;
} __fncsele_bits;

/*PDENE*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32 PD3  : 1;
  __REG32 PD4  : 1;
  __REG32      :27;
} __pdene_bits;

/*PUENE*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32 PU3  : 1;
  __REG32 PU4  : 1;
  __REG32      :27;
} __puene_bits;

/*FNCSELF*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32 FS3  : 2;
  __REG32      :24;
} __fncself_bits;

/*PDENF*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32 PD3  : 1;
  __REG32      :28;
} __pdenf_bits;

/*PUENF*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32 PU3  : 1;
  __REG32      :28;
} __puenf_bits;

/*FNCSELG*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32 FS3  : 2;
  __REG32 FS4  : 2;
  __REG32 FS5  : 2;
  __REG32 FS6  : 2;
  __REG32      :18;
} __fncselg_bits;

/*PDENG*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32 PD3  : 1;
  __REG32 PD4  : 1;
  __REG32 PD5  : 1;
  __REG32 PD6  : 1;
  __REG32      :25;
} __pdeng_bits;

/*PUENG*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32 PU3  : 1;
  __REG32 PU4  : 1;
  __REG32 PU5  : 1;
  __REG32 PU6  : 1;
  __REG32      :25;
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

/*FNCSELK*/
typedef struct {
  __REG32 FS0  : 2;
  __REG32 FS1  : 2;
  __REG32 FS2  : 2;
  __REG32 FS3  : 2;
  __REG32 FS4  : 2;
  __REG32 FS5  : 2;
  __REG32      :20;
} __fncselk_bits;

/*PDENK*/
typedef struct {
  __REG32 PD0  : 1;
  __REG32 PD1  : 1;
  __REG32 PD2  : 1;
  __REG32 PD3  : 1;
  __REG32 PD4  : 1;
  __REG32 PD5  : 1;
  __REG32      :26;
} __pdenk_bits;

/*PUENK*/
typedef struct {
  __REG32 PU0  : 1;
  __REG32 PU1  : 1;
  __REG32 PU2  : 1;
  __REG32 PU3  : 1;
  __REG32 PU4  : 1;
  __REG32 PU5  : 1;
  __REG32      :26;
} __puenk_bits;

/*SSOCTL*/
typedef struct {
  __REG32 SSO00 : 1;
  __REG32 SSO01 : 1;
  __REG32 SSO02 : 1;
  __REG32       : 1;
  __REG32 SSO1  : 1;
  __REG32       :27;
} __ssoctl_bits;

/*CANM*/
typedef struct {
  __REG32 MOD  : 2;
  __REG32      :30;
} __canm_bits;

/*SIOCKEN*/
typedef struct {
  __REG32 SCK0EN  : 1;
  __REG32 SCK1EN  : 1;
  __REG32 SCK2EN  : 1;
  __REG32     		:29;
} __siocken_bits;

/*ADCCTL*/
typedef struct {
  __REG32 UVWS0   : 2;
  __REG32 UVWS1   : 2;
  __REG32 UVWS2   : 2;
  __REG32 UVWS3   : 2;
  __REG32         : 8;
  __REG32 SEL1    : 1;
  __REG32 SEL2    : 1;
  __REG32     		:14;
} __adcctl_bits;

/*CAL_TRIG_EN*/
typedef struct {
  __REG32 CALEN   : 1;
  __REG32     		:31;
} __cal_trig_en_bits;

/*MONA*/
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
  __REG32 O7   : 1;
  __REG32      :24;
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
  __REG32 OE7  : 1;
  __REG32      :24;
} __oena_bits;

/*MONB*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32 I3   : 1;
  __REG32 I4   : 1;
  __REG32      :27;
} __monb_bits;

/*OUTB*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32 O3   : 1;
  __REG32 O4   : 1;
  __REG32      :27;
} __outb_bits;

/*OENB*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32 OE3  : 1;
  __REG32 OE4  : 1;
  __REG32      :27;
} __oenb_bits;

/*MONC*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32 I3   : 1;
  __REG32 I4   : 1;
  __REG32 I5   : 1;
  __REG32 I6   : 1;
  __REG32      :25;
} __monc_bits;

/*OUTC*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32 O3   : 1;
  __REG32 O4   : 1;
  __REG32 O5   : 1;
  __REG32 O6   : 1;
  __REG32      :25;
} __outc_bits;

/*OENC*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32 OE3  : 1;
  __REG32 OE4  : 1;
  __REG32 OE5  : 1;
  __REG32 OE6  : 1;
  __REG32      :25;
} __oenc_bits;

/*MOND*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32 I3   : 1;
  __REG32 I4   : 1;
  __REG32 I5   : 1;
  __REG32      :26;
} __mond_bits;

/*OUTD*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32 O3   : 1;
  __REG32 O4   : 1;
  __REG32 O5   : 1;
  __REG32      :26;
} __outd_bits;

/*OEND*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32 OE3  : 1;
  __REG32 OE4  : 1;
  __REG32 OE5  : 1;
  __REG32      :26;
} __oend_bits;

/*MONE*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32 I3   : 1;
  __REG32 I4   : 1;
  __REG32      :27;
} __mone_bits;

/*OUTE*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32 O3   : 1;
  __REG32 O4   : 1;
  __REG32      :27;
} __oute_bits;

/*OENE*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32 OE3  : 1;
  __REG32 OE4  : 1;
  __REG32      :27;
} __oene_bits;

/*MONF*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32 I3   : 1;
  __REG32      :28;
} __monf_bits;

/*OUTF*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32 O3   : 1;
  __REG32      :28;
} __outf_bits;

/*OENF*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32 OE3  : 1;
  __REG32      :28;
} __oenf_bits;

/*MONG*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32 I3   : 1;
  __REG32 I4   : 1;
  __REG32 I5   : 1;
  __REG32 I6   : 1;
  __REG32      :25;
} __mong_bits;

/*OUTG*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32 O3   : 1;
  __REG32 O4   : 1;
  __REG32 O5   : 1;
  __REG32 O6   : 1;
  __REG32      :25;
} __outg_bits;

/*OENG*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32 OE3  : 1;
  __REG32 OE4  : 1;
  __REG32 OE5  : 1;
  __REG32 OE6  : 1;
  __REG32      :25;
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
  __REG32 I2   : 1;
  __REG32 I3   : 1;
  __REG32 I4   : 1;
  __REG32 I5   : 1;
  __REG32      :26;
} __monk_bits;

/*OUTK*/
typedef struct {
  __REG32 O0   : 1;
  __REG32 O1   : 1;
  __REG32 O2   : 1;
  __REG32 O3   : 1;
  __REG32 O4   : 1;
  __REG32 O5   : 1;
  __REG32      :26;
} __outk_bits;

/*OENK*/
typedef struct {
  __REG32 OE0  : 1;
  __REG32 OE1  : 1;
  __REG32 OE2  : 1;
  __REG32 OE3  : 1;
  __REG32 OE4  : 1;
  __REG32 OE5  : 1;
  __REG32      :26;
} __oenk_bits;

/*MONM*/
typedef struct {
  __REG32 I0   : 1;
  __REG32 I1   : 1;
  __REG32 I2   : 1;
  __REG32      :29;
} __monm_bits;

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

/*DMA Transfer Type Register (DMAXFTYP)*/
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

/*DMA Transfer Number of Descriptor Register (DSNUM)*/
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

/*TBT RUN Register (TCMPx_TBTRUN)*/
typedef struct {
  __REG32  TBRUN    : 1;
  __REG32  DIVRUN   : 1;
  __REG32           : 2;
  __REG32  TBPSEN   : 1;
  __REG32           :27;
} __tcmp_tbtrun_bits;

/*TBT Control Register (TCMPx_TBTCR)*/
typedef struct {
  __REG32           : 4;
  __REG32  DIVSEL   : 7;
  __REG32           :21;
} __tcmp_tbtcr_bits;

/*CMP Control Register (TCMPx_CMPyCR)*/
typedef struct {
  __REG32  CMPEN	  : 1;
  __REG32  CMPDBEN  : 1;
  __REG32  CNTCLEN  : 1;
  __REG32           :29;
} __tcmp_cmpcr_bits;

/*CMP Output Level Register (TCMPx_CMPDOy)*/
typedef struct {
  __REG32  DOR   	  : 1;
  __REG32           :31;
} __tcmp_cmpdo_bits;

/*CMP Output Status Register (TCMPx_CMPSDy)*/
typedef struct {
  __REG32  LGTO   	: 1;
  __REG32           :31;
} __tcmp_cmpsd_bits;

/*CMP Output Mode Register (TCMPx_CMPOMy)*/
typedef struct {
  __REG32  DOM    	: 1;
  __REG32           :31;
} __tcmp_cmpom_bits;

/*TBT RUN Register (TCAPx_TBTRUN)*/
typedef struct {
  __REG32  TBTRUN   : 1;
  __REG32  DIVRUN   : 1;
  __REG32           : 2;
  __REG32  TBTPSEN  : 1;
  __REG32           :27;
} __tcap_tbtrun_bits;

/*TBT Control Register (TCAPx_TBTCR)*/
typedef struct {
  __REG32           : 4;
  __REG32  DIVSEL   : 7;
  __REG32           :21;
} __tcap_tbtcr_bits;

/*Input Capture Control Register (TCAP0_CAPCR)*/
typedef struct {
  __REG32  CAPNF0   : 1;
  __REG32  CAPNF1   : 1;
  __REG32  CAPNF2   : 1;
  __REG32  CAPNF3   : 1;
  __REG32  CAPNF4   : 1;
  __REG32  CAPNF5   : 1;
  __REG32           :26;
} __tcap_incapcr_bits;

/*Compare Control Register (TCAP0_CMPxCR)*/
typedef struct {
  __REG32  CMPEN    : 1;
  __REG32  CMPDBEN  : 1;
  __REG32  CNTCLEN  : 1;
  __REG32           :29;
} __tcap_cmpcr_bits;

/*PWM Sync Control Register (PWMSYNCRUN)*/
typedef struct {
  __REG32  R0  		  : 1;
  __REG32  PR0  		: 1;
  __REG32  R1  		  : 1;
  __REG32  PR1  		: 1;
  __REG32  R2  		  : 1;
  __REG32  PR2  		: 1;
  __REG32  R3  		  : 1;
  __REG32  PR3  		: 1;
  __REG32  					:24;
} __pwmsyncrun_bits;

/*PWM RUN Register (PWMn_RUN)*/
typedef struct {
  __REG32  TBRUN    : 1;
  __REG32           : 1;
  __REG32  TBPRUN   : 1;
  __REG32           : 1;
  __REG32  TBPSEN   : 1;
  __REG32           :27;
} __pwm_run_bits;

/*PWM Control Register PWMn_CR*/
typedef struct {
  __REG32  TBWBUF   : 1;
  __REG32           :31;
} __pwm_cr_bits;

/*PWM Mode Register PWMn_MOD*/
typedef struct {
  __REG32  DIVSEL   : 3;
  __REG32           :29;
} __pwm_mod_bits;

/*PWM Output Polarity Register PWMn_OUTCTRL*/
typedef struct {
  __REG32  TBACT    : 1;
  __REG32           :31;
} __pwm_outctrl_bits;

/*PWM Period Register PWMn_PRICMP*/
typedef struct {
  __REG32  TBRG0    :24;
  __REG32           : 8;
} __pwm_rg0_bits;

/*PWM Period Register PWMn_DUTYCMP*/
typedef struct {
  __REG32  TBRG1    :24;
  __REG32           : 8;
} __pwm_rg1_bits;

/*PWM Counter Register PWMn_CNT*/
typedef struct {
  __REG32  CNT      :24;
  __REG32           : 8;
} __pwm_cnt_bits;

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
  __REG8 RB8      : 1;
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

/*SCxBRCR*/
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

/* CAN Message Identifier (ID0 .. ID3) */
typedef struct {
  __REG32  ID             :29;
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

/* CAN Acceptance ID Mask Register (MBAM) */
typedef struct {
  __REG32  AM              :29;
  __REG32                  : 2;
  __REG32  AMI             : 1;
} __canmbam_bits;

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
  __REG32  INTLB           : 1;
  __REG32  SUR             : 1;
  __REG32                  :20;
} __canmcr_bits;

/* CAN Bit Configuration Register 1 (BCR1) */
typedef struct {
  __REG32  BRP             :10;
  __REG32                  :22;
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

/*ESEI MCR*/
typedef struct {
  __REG32  BCLR    	: 1;
  __REG32  STOP   	: 1;
  __REG32  LOOP   	: 1;
  __REG32           : 3;
  __REG32  OPMODE 	: 2;
  __REG32           :24;
} __esei_mcr_bits;

/*ESEI CR0*/
typedef struct {
  __REG32  SPOL    	: 1;
  __REG32  SPHA	  	: 1;
  __REG32  SBOS   	: 1;
  __REG32  MSTR     : 1;
  __REG32  IFSPSE 	: 1;
  __REG32  SSIVAL   : 1;
  __REG32           : 2;
  __REG32  STFIE	 	: 1;
  __REG32  SUEIE	 	: 1;
  __REG32  SOEIE	 	: 1;
  __REG32  SILIE	 	: 1;
  __REG32           :20;
} __esei_cr0_bits;

/*ESEI CR1*/
typedef struct {
  __REG32  SSZ     	: 5;
  __REG32           : 3;
  __REG32  SER		 	: 8;
  __REG32           :16;
} __esei_cr1_bits;

/*ESEI FSR*/
typedef struct {
  __REG32  IFS     	:10;
  __REG32           :22;
} __esei_fsr_bits;

/*ESEI SSR*/
typedef struct {
  __REG32  SSS     	: 8;
  __REG32           :24;
} __esei_ssr_bits;

/*ESEI SR*/
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
} __esei_sr_bits;

/*ESEI DR*/
typedef struct {
  __REG32 DATA   	  :16;
  __REG32           :16;
} __esei_dr_bits;

/*ESEI RSR*/
typedef struct {
  __REG32  RDATA	 	:16;
  __REG32           :16;
} __esei_rsr_bits;

/*ESEI FLR*/
typedef struct {
  __REG32  SRBFL  	: 5;
  __REG32           : 3;
  __REG32  STBFL  	: 5;
  __REG32           :19;
} __esei_flr_bits;

/*ESEI ILR*/
typedef struct {
  __REG32  RXIFL  	: 5;
  __REG32           : 3;
  __REG32  TXIFL  	: 5;
  __REG32           :19;
} __esei_ilr_bits;

/*ESEI PR*/
typedef struct {
  __REG32  SEISE  	: 1;
  __REG32  SEIE   	: 1;
  __REG32  SEEO   	: 1;
  __REG32           : 1;
  __REG32  SEEN   	: 1;
  __REG32  SEP01   	: 2;
  __REG32           :25;
} __esei_pr_bits;

/*ESEI LCR*/
typedef struct {
  __REG32  SLB	  	: 1;
  __REG32  SLTB   	: 1;
  __REG32           :30;
} __esei_lcr_bits;

/*ESEI DER*/
typedef union {
  /*ESEIx_DER*/
  struct {
    __REG32  SBD	  	:16;
    __REG32  SFL	  	: 5;
    __REG32  SCID	  	: 3;
    __REG32  SRBFL  	: 5;
    __REG32           : 2;
    __REG32  PR       : 1;
  };
  /*ESEIx_DERW*/
  struct {
    __REG32  SBD	  	:16;
    __REG32  SFL	  	: 5;
    __REG32  SCID	  	: 3;
    __REG32  SSST    	: 1;
    __REG32           : 7;
  } __write;
} __esei_der_bits;

/*ESEI EICR*/
typedef struct {
  __REG32  EIC	  	: 1;
  __REG32           :31;
} __esei_eicr_bits;

/*ESEI RICR*/
typedef struct {
  __REG32  RIC	  	: 1;
  __REG32           :31;
} __esei_ricr_bits;

/*ESEI TICR*/
typedef struct {
  __REG32  TIC	  	: 1;
  __REG32           :31;
} __esei_ticr_bits;

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
typedef struct {
  __REG32  VAL		  :15;
  __REG32           :17;
} __insdat_bits;

/*VE Control Registers*/
typedef struct {
  __REG32 VEEN         : 1;
  __REG32              :31;
} __veen_bits;

/*VE Control Registers*/
typedef struct {
  __REG32 VCPURT       : 1;
  __REG32              :31;
} __cpuruntrg_bits;

/*TASKAPP Register*/
typedef struct {
  __REG32 VTASK        : 4;
  __REG32              :28;
} __taskapp_bits;

/*ACTSCH Register*/
typedef struct {
  __REG32 VACT         : 4;
  __REG32              :28;
} __actsch_bits;

/*REPTIME Register*/
typedef struct {
  __REG32 VREP         : 4;
  __REG32              :28;
} __reptime_bits;

/*TRGMODE Register*/
typedef struct {
  __REG32 VADCTRG      : 4;
  __REG32              :28;
} __trgmode_bits;

/*ERRINTEN Register*/
typedef struct {
  __REG32 VERREN       : 1;
  __REG32              :31;
} __errinten_bits;

/*COMPEND Register*/
typedef struct {
  __REG32 VCEND        : 1;
  __REG32              :31;
} __compend_bits;

/*ERRDET Register*/
typedef struct {
  __REG32 VERRD        : 1;
  __REG32              :31;
} __errdet_bits;

/*SCHTASKRUN Register*/
typedef struct {
  __REG32 VRSCH        : 1;
  __REG32              : 3;
  __REG32 VRTASK       : 8;
  __REG32              :20;
} __schtaskrun_bits;

/*MCTLF Register*/
typedef struct {
  __REG32              : 2;
  __REG32 LVTF         : 1;
  __REG32              : 1;
  __REG32 PLSLF        : 1;
  __REG32 PLSLFM       : 1;
  __REG32              :26;
} __mctlfx_bits;

/*MODE Register*/
typedef struct {
  __REG32 PVIEN        : 1;
  __REG32 ZIEN         : 1;
  __REG32 OCRMD        : 2;
  __REG32              :28;
} __modex_bits;

/*FMODE Register*/
typedef struct {
  __REG32 C2PEN        : 1;
  __REG32              : 1;
  __REG32 IDMODE       : 2;
  __REG32              : 5;
  __REG32 MREGDIS      : 1;
  __REG32              :22;
} __fmodex_bits;

/*TPWM Register*/
typedef struct {
  __REG32 TPWM         :16;
  __REG32              :16;
} __tpwmx_bits;

/*OMEGA Register*/
typedef struct {
  __REG32 OMEGA        :16;
  __REG32              :16;
} __omegax_bits;

/*THETA Register*/
typedef struct {
  __REG32 THETA        :16;
  __REG32              :16;
} __thetax_bits;

/*IDREF Register*/
typedef struct {
  __REG32 IDREF        :16;
  __REG32              :16;
} __idrefx_bits;

/*IQREF Register*/
typedef struct {
  __REG32 IQREF        :16;
  __REG32              :16;
} __iqrefx_bits;

/*VD Register*/
typedef struct {
  __REG32 VD           :32;
} __vdx_bits;

/*VQ Register*/
typedef struct {
  __REG32 VQ           :32;
} __vqx_bits;

/*CIDKI Register*/
typedef struct {
  __REG32 CIDKI        :16;
  __REG32              :16;
} __cidkix_bits;

/*CIDKP Register*/
typedef struct {
  __REG32 CIDKP        :16;
  __REG32              :16;
} __cidkpx_bits;

/*CIQKI Register*/
typedef struct {
  __REG32 CIQKI        :16;
  __REG32              :16;
} __ciqkix_bits;

/*CIQKP Register*/
typedef struct {
  __REG32 CIQKP        :16;
  __REG32              :16;
} __ciqkpx_bits;

/*VDIH Register*/
typedef struct {
  __REG32 VDIH         :32;
} __vdihx_bits;

/*VDILH Register*/
typedef struct {
  __REG32              :16;
  __REG32 VDILH        :16;
} __vdilhx_bits;

/*VQIH Register*/
typedef struct {
  __REG32 VQIH         :32;
} __vqihx_bits;

/*VQILH Register*/
typedef struct {
  __REG32              :16;
  __REG32 VQILH        :16;
} __vqilhx_bits;

/*FPWMCHG Register*/
typedef struct {
  __REG32 FPWMCHG      :16;
  __REG32              :16;
} __fpwmchgx_bits;

/*PWM Period Register*/
typedef struct {
  __REG32 VMDPRD       :16;
  __REG32              :16;
} __vmdprdx_bits;

/*MINPLS Register*/
typedef struct {
  __REG32 MINPLS       :16;
  __REG32              :16;
} __minplsx_bits;

/*TRGCRC Register*/
typedef struct {
  __REG32 TRGCRC       :16;
  __REG32              :16;
} __trgcrcx_bits;

/*COS Registers*/
typedef struct {
  __REG32 COS          :16;
  __REG32              :16;
} __cosx_bits;

/*SIN Registers*/
typedef struct {
  __REG32 SIN          :16;
  __REG32              :16;
} __sinx_bits;

/*COSM Registers*/
typedef struct {
  __REG32 COSM         :16;
  __REG32              :16;
} __cosmx_bits;

/*SINM Registers*/
typedef struct {
  __REG32 SINM         :16;
  __REG32              :16;
} __sinmx_bits;

/*Sector Register*/
typedef struct {
  __REG32 SECTOR       : 4;
  __REG32              :28;
} __sectorx_bits;

/*Sector Register*/
typedef struct {
  __REG32 SECTORM      : 4;
  __REG32              :28;
} __sectormx_bits;

/*Zero-Current Registers*/
typedef struct {
  __REG32 IAO          :16;
  __REG32              :16;
} __iaox_bits;

/*Zero-Current Registers*/
typedef struct {
  __REG32 IBO          :16;
  __REG32              :16;
} __ibox_bits;

/*Zero-Current Registers*/
typedef struct {
  __REG32 ICO          :16;
  __REG32              :16;
} __icox_bits;

/*Current ADC Result Registers*/
typedef struct {
  __REG32 IAADC        :16;
  __REG32              :16;
} __iaadcx_bits;

/*Current ADC Result Registers*/
typedef struct {
  __REG32 IBADC        :16;
  __REG32              :16;
} __ibadcx_bits;

/*Current ADC Result Registers*/
typedef struct {
  __REG32 ICADC        :16;
  __REG32              :16;
} __icadcx_bits;

/*Supply Voltage Register*/
typedef struct {
  __REG32 VDC          :16;
  __REG32              :16;
} __vdcx_bits;

/*dq Current Registers*/
typedef struct {
  __REG32 ID           :32;
} __idx_bits;

/*dq Current Registers*/
typedef struct {
  __REG32 IQ           :32;
} __iqx_bits;     

/*PWM Duty Register U Phase*/
typedef struct {
  __REG32 VCMPU        :16;
  __REG32              :16;
} __vcmpux_bits;

/*PWM Duty Register V Phase*/
typedef struct {
  __REG32 VCMPV        :16;
  __REG32              :16;
} __vcmpvx_bits;

/*PWM Duty Register W Phase*/
typedef struct {
  __REG32 VCMPW        :16;
  __REG32              :16;
} __vcmpwx_bits;

/*6-Phase Output Control Register*/
typedef struct {
  __REG32 UOC          : 2;
  __REG32 VOC          : 2;
  __REG32 WOC          : 2;
  __REG32              : 2;
  __REG32 UPWM         : 1;
  __REG32 VPWM         : 1;
  __REG32 WPWM         : 1;
  __REG32              :21;
} __outcrx_bits;

/*VTRGCMP Register 0*/
typedef struct {
  __REG32 VTRGCMP      :16;
  __REG32              :16;
} __vtrgcmpax_bits;

/*VTRGCMP Register 1*/
typedef struct {
  __REG32 VTRGCMP      :16;
  __REG32              :16;
} __vtrgcmpbx_bits;

/*EMGRS Register*/
typedef struct {
  __REG32 EMGRS        : 1;
  __REG32              :31;
} __emgrsx_bits;

/*TSKPGMx Registers (x=0-F)*/
typedef struct {
  __REG32 CODE         : 8;
  __REG32              : 1;
  __REG32 ADC          : 1;
  __REG32 END          : 1;
  __REG32 PRDST        : 1;
  __REG32 PRDED        : 1;
  __REG32              :19;
} __tskpgmx_bits;

/*TSKPGMx Registers (x=0-F)*/
typedef struct {
  __REG32 PTR          : 4;
  __REG32              :28;
} __tskinitp_bits;

/*PMD RATECR0 (Rate Counter 0 Control Register)*/
typedef struct {
  __REG32 RC_EN0       : 1;
  __REG32              :31;
} __pmd_ratecr0_bits;

/*PMD RCCCR0 (Rate Counter 0 Compensation Control Register)*/
typedef struct {
  __REG32 RCCEN0       : 1;
  __REG32              :31;
} __pmd_rcccr0_bits;

/*PMD RATETSEL0 (RATE0 Register Update Timing Select Register)*/
typedef struct {
  __REG32 RUTS0        : 3;
  __REG32              :29;
} __pmd_ratetsel0_bits;

/*PMD RATE0 (Rate Counter 0 Carrier Frequency Fine-Tuning Register)*/
typedef struct {
  __REG32 RATE0        :16;
  __REG32              :16;
} __pmd_rate0_bits;

/*PMD CARSET0 (Carrier 0 Reference Frequency Set Register)*/
typedef struct {
  __REG32 CAR0         : 4;
  __REG32 CNT0         : 4;
  __REG32 INTAS0       : 2;
  __REG32 INTBS0       : 3;
  __REG32 CIUTS0       : 3;
  __REG32              :16;
} __pmd_carset0_bits;

/*PMD CARSIF0 (Carrier 0 Phase Set Register)*/
typedef struct {
  __REG32 CARSIF0      :16;
  __REG32              :16;
} __pmd_carsif0_bits;

/*PMD CMPU0 (Triangle Wave PWM U-Phase Voltage Instruction Value Register)*/
typedef struct {
  __REG32 CMPU0        :16;
  __REG32              :16;
} __pmd_cmpu0_bits;

/*PMD CMPV0 (Triangle Wave PWM V-Phase Voltage Instruction Value Register)*/
typedef struct {
  __REG32 CMPV0        :16;
  __REG32              :16;
} __pmd_cmpv0_bits;

/*PMD CMPW0 (Triangle Wave PWM W-Phase Voltage Instruction Value Register)*/
typedef struct {
  __REG32 CMPW0        :16;
  __REG32              :16;
} __pmd_cmpw0_bits;

/*PMD NOKORD0 (Carrier 0 Sawtooth Wave Read Register)*/
typedef struct {
  __REG32 NOKORD0      :16;
  __REG32              :16;
} __pmd_nokord0_bits;

/*PMD CARH0 (Carrier 0 Counter Register)*/
typedef struct {
  __REG32 CARH0        : 4;
  __REG32              :28;
} __pmd_carh0_bits;

/*PMD CARL0 (Carrier 0 Read Register)*/
typedef struct {
  __REG32 CARL0        :16;
  __REG32              :16;
} __pmd_carl0_bits;

/*PMD CARSET1 (Carrier 1 Reference Frequency Set Register)*/
typedef struct {
  __REG32 CAR1         : 4;
  __REG32 CNT1         : 4;
  __REG32 INTS1        : 2;
  __REG32              :22;
} __pmd_carset1_bits;

/*PMD CARSIF1 (Carrier 1 Phase Set Register)*/
typedef struct {
  __REG32 CARSIF1      :16;
  __REG32              :16;
} __pmd_carsif1_bits;

/*PMD TRGCMPA1 (Carrier 1 Trigger Compare Register A)*/
typedef struct {
  __REG32 TRGCMPA1     :16;
  __REG32              :16;
} __pmd_trgcmpa1_bits;

/*PMD TRGCMPB1 (Carrier 1 Trigger Compare Register B)*/
typedef struct {
  __REG32 TRGCMPB1     :16;
  __REG32              :16;
} __pmd_trgcmpb1_bits;

/*PMD NOKORD1 (Carrier 1 Sawtooth Wave Read Register)*/
typedef struct {
  __REG32 NOKORD1      :16;
  __REG32              :16;
} __pmd_nokord1_bits;

/*PMD CARH1 (Carrier 1 Counter Register)*/
typedef struct {
  __REG32 CARH1        : 4;
  __REG32              :28;
} __pmd_carh1_bits;

/*PMD RATECR1 (Rate Counter 1 Control Register)*/
typedef struct {
  __REG32 RC_EN1       : 1;
  __REG32              :31;
} __pmd_ratecr1_bits;

/*PMD RCCCR1 (Rate Counter 1 Compensation Control Register)*/
typedef struct {
  __REG32 RCCEN1       : 1;
  __REG32              :31;
} __pmd_rcccr1_bits;

/*PMD RATETSEL1 (RATE1 Register Update Timing Select Register)*/
typedef struct {
  __REG32 RUTS1        : 3;
  __REG32              :29;
} __pmd_ratetsel1_bits;

/*PMD RATE1 (Rate Counter 1 Carrier Frequency Fine-Tuning Register)*/
typedef struct {
  __REG32 RATE1        :16;
  __REG32              :16;
} __pmd_rate1_bits;

/*PMD CARSET2 (Carrier 2 Reference Frequency Set Register)*/
typedef struct {
  __REG32 CAR2         : 4;
  __REG32 CNT2         : 4;
  __REG32 INTS2        : 2;
  __REG32              :22;
} __pmd_carset2_bits;

/*PMD CARSIF2 (Carrier 2 Phase Set Register)*/
typedef struct {
  __REG32 CARSIF2      :16;
  __REG32              :16;
} __pmd_carsif2_bits;

/*PMD CPWMA2 (Carrier 2 Compare Register A)*/
typedef struct {
  __REG32 CPWMA2       :16;
  __REG32              :16;
} __pmd_cpwma2_bits;

/*PMD CPWMB2 (Carrier 2 Compare Register B)*/
typedef struct {
  __REG32 CPWMB2       :16;
  __REG32              :16;
} __pmd_cpwmb2_bits;

/*PMD NOKORD2 (Carrier 2 Sawtooth Wave Read Register)*/
typedef struct {
  __REG32 NOKORD2      :16;
  __REG32              :16;
} __pmd_nokord2_bits;

/*PMD CARH2 (Carrier 2 Counter Register)*/
typedef struct {
  __REG32 CARH2        : 4;
  __REG32              :28;
} __pmd_carh2_bits;

/*PMD CARL2 (Carrier 2 Read Register)*/
typedef struct {
  __REG32 CARL2        :16;
  __REG32              :16;
} __pmd_carl2_bits;

/*PMD CARCNT (CARSET Update Timing Set Register)*/
typedef struct {
  __REG32 CARSET0      : 2;
  __REG32              : 2;
  __REG32 CARSET1      : 2;
  __REG32              : 2;
  __REG32 CARSET2      : 2;
  __REG32              :22;
} __pmd_carcnt_bits;

/*PMD SIFCNT (CARSIF Update Timing Set Register)*/
typedef struct {
  __REG32 CARSIF0      : 3;
  __REG32              : 1;
  __REG32 CARSIF2      : 3;
  __REG32              : 1;
  __REG32 CARSIF3      : 3;
  __REG32              :21;
} __pmd_sifcnt_bits;

/*PMD CMPCNT (Instruction Value Update Timing Set Register)*/
typedef struct {
  __REG32 UVWUTS       : 3;
  __REG32              : 1;
  __REG32 TCAUTS       : 2;
  __REG32 TCBUTS       : 2;
  __REG32 CPAUTS       : 2;
  __REG32 CPBUTS       : 2;
  __REG32              :20;
} __pmd_cmpcnt_bits;

/*PMD PO_DTR (PO Dead Time Set Register)*/
typedef struct {
  __REG32 DTR          :12;
  __REG32              :20;
} __pmd_po_dtr_bits;

/*PMD PO_DTR (PO Dead Time Set Register)*/
typedef struct {
  __REG32 MPR          :12;
  __REG32              :20;
} __pmd_po_mpr_bits;

/*PMD PO_MDEN (PO Output Enable Register)*/
typedef struct {
  __REG32 PWMEN        : 1;
  __REG32              :31;
} __pmd_po_mden_bits;

/*PMD PO_PORTMD (PO Port Output Mode Register)*/
typedef struct {
  __REG32 PORTMD       : 2;
  __REG32              :30;
} __pmd_po_portmd_bits;

/*PMD PO_MDCR (PO Output Control Register)*/
typedef struct {
  __REG32              : 4;
  __REG32 DTYMD        : 1;
  __REG32 SYNTMD       : 1;
  __REG32              :26;
} __pmd_po_mdcr_bits;

/*PMD PO_MDOUT (PO Output Waveform Control Register)*/
typedef struct {
  __REG32 UOC          : 2;
  __REG32 VOC          : 2;
  __REG32 WOC          : 2;
  __REG32              : 2;
  __REG32 UPWM         : 1;
  __REG32 VPWM         : 1;
  __REG32 WPWM         : 1;
  __REG32              :21;
} __pmd_po_mdout_bits;

/*PMD PO_MDPOT (PO Output Set Register)*/
typedef struct {
  __REG32 PSYNCS       : 2;
  __REG32 POLL         : 1;
  __REG32 POLH         : 1;
  __REG32              :28;
} __pmd_po_mdpot_bits;

/*PMD PO_EMGREL (PO EMG Release Register)*/
typedef struct {
  __REG32 EMGREL       : 8;
  __REG32              :24;
} __pmd_po_emgrel_bits;

/*PMD PO_EMGCR (PO EMG Control Register)*/
typedef struct {
  __REG32 EMGEN        : 1;
  __REG32 EMGRS        : 1;
  __REG32              : 1;
  __REG32 EMGMD        : 2;
  __REG32 INHEN        : 1;
  __REG32              : 2;
  __REG32 EMGCNT       : 4;
  __REG32              :20;
} __pmd_po_emgcr_bits;

/*PMD PO_EMGSTA (PO EMG Status Register)*/
typedef struct {
  __REG32 EMGST        : 1;
  __REG32 EMGI         : 1;
  __REG32 U_MP         : 1;
  __REG32 X_MP         : 1;
  __REG32 POU_OUT      : 1;
  __REG32 POX_OUT      : 1;
  __REG32 V_MP         : 1;
  __REG32 Y_MP         : 1;
  __REG32 POV_OUT      : 1;
  __REG32 POY_OUT      : 1;
  __REG32 W_MP         : 1;
  __REG32 Z_MP         : 1;
  __REG32 POW_OUT      : 1;
  __REG32 POZ_OUT      : 1;
  __REG32              :18;
} __pmd_po_emgsta_bits;

/*PMD MSET (QR Pole Number Set Register)*/
typedef struct {
  __REG32 MP           : 4;
  __REG32 SP           : 3;
  __REG32              :25;
} __pmd_mset_bits;

/*PMD WLOAD (Register Update Timing Set Register)*/
typedef struct {
  __REG32 WL           : 3;
  __REG32              : 1;
  __REG32 WQAM         : 3;
  __REG32              : 1;
  __REG32 Q1ATS        : 2;
  __REG32 Q1BTS        : 2;
  __REG32              :20;
} __pmd_wload_bits;

/*PMD QSH (Slip Angle Frequency Integral Value (H))*/
typedef struct {
  __REG32 QSH          : 4;
  __REG32              :28;
} __pmd_qsh_bits;

/*PMD QT (Torque Angle Set Register)*/
typedef struct {
  __REG32 QT           :16;
  __REG32              :16;
} __pmd_qt_bits;

/*PMD SPWMQTU (U-Phase Torque Angle Adjustment Register)*/
typedef struct {
  __REG32 SPWMQTU      :16;
  __REG32              :16;
} __pmd_spwmqtu_bits;

/*PMD SPWMQTV (V-Phase Torque Angle Adjustment Register)*/
typedef struct {
  __REG32 SPWMQTV      :16;
  __REG32              :16;
} __pmd_spwmqtv_bits;

/*PMD SPWMQTW (W-Phase Torque Angle Adjustment Register)*/
typedef struct {
  __REG32 SPWMQTW      :16;
  __REG32              :16;
} __pmd_spwmqtw_bits;

/*PMD Q1A ( Read Register A)*/
typedef struct {
  __REG32 Q1A          :16;
  __REG32              :16;
} __pmd_q1a_bits;

/*PMD Q1B ( Read Register B)*/
typedef struct {
  __REG32 Q1B          :16;
  __REG32              :16;
} __pmd_q1b_bits;

/*PMD Q1C ( Read Register C)*/
typedef struct {
  __REG32 Q1C          :16;
  __REG32              :16;
} __pmd_q1c_bits;

/*PMD QRWDT (QR Preset Value Register)*/
typedef struct {
  __REG32 QRWDT        :20;
  __REG32              :12;
} __pmd_qrwdt_bits;

/*PMD QRCR (ABZ Decoder Control Register)*/
typedef struct {
  __REG32 PULSE        : 4;
  __REG32 CLRVAL       : 3;
  __REG32 CLREN        : 1;
  __REG32 ZUEEN        : 1;
  __REG32              :23;
} __pmd_qrcr_bits;

/*PMD QRSW (ABZ Decode Select Register)*/
typedef struct {
  __REG32 SWEMA        : 1;
  __REG32 SWQAD        : 1;
  __REG32              :30;
} __pmd_qrsw_bits;

/*PMD QROFFSET (Resolver Error Adjustment Offset Register)*/
typedef struct {
  __REG32 OFFSET       :16;
  __REG32              :16;
} __pmd_qroffset_bits;

/*PMD QRCMP (QR Compare Value Set Register)*/
typedef struct {
  __REG32 QRCMP        :16;
  __REG32              :16;
} __pmd_qrcmp_bits;

/*PMD QRADD (QRCMP Add Value Set Register)*/
typedef struct {
  __REG32 QRADD        :16;
  __REG32              :16;
} __pmd_qradd_bits;

/*PMD QRCOUNT (QR Read Register)*/
typedef struct {
  __REG32 QRCOUNT      :16;
  __REG32              :16;
} __pmd_qrcount_bits;

/*PMD QROUT (QR Read Register (After adding QROFFSET))*/
typedef struct {
  __REG32 QROUT        :20;
  __REG32              :12;
} __pmd_qrout_bits;

/*PMD THETAAD (AD Trigger Sync Rotation Angle Register)*/
typedef struct {
  __REG32 THETAAD      :20;
  __REG32              :12;
} __pmd_thetaad_bits;

/*PMD THETAOUT (AD Trigger Sync Post Select Rotation Angle Register)*/
typedef struct {
  __REG32 THETAOUT     :20;
  __REG32              :12;
} __pmd_thetaout_bits;

/*PMD QRWCR (QRWDT Preset Control Register)*/
typedef struct {
  __REG32 PRESET       : 1;
  __REG32              :31;
} __pmd_qrwcr_bits;

/*PMD DPWMTSEL (Direct PWM Update Timing Select Register)*/
typedef struct {
  __REG32 DPUTS        : 3;
  __REG32              :29;
} __pmd_dpwmtsel_bits;

/*PMD DIRPWM (Direct PWM Register)*/
typedef struct {
  __REG32 DIR_U        : 1;
  __REG32 DIR_V        : 1;
  __REG32 DIR_W        : 1;
  __REG32              :29;
} __pmd_dirpwm_bits;

/*PMD VESW (VE Toggle Register)*/
typedef struct {
  __REG32 VE0          : 1;
  __REG32              : 4;
  __REG32 VE5          : 1;
  __REG32 VE6          : 1;
  __REG32 VE7          : 1;
  __REG32              :24;
} __pmd_vesw_bits;

/*PMD PO_DSW (PO PWM Toggle Register)*/
typedef struct {
  __REG32 SWSPU        : 1;
  __REG32 SWSPV        : 1;
  __REG32 SWSPW        : 1;
  __REG32              : 1;
  __REG32 SWDPU        : 1;
  __REG32 SWDPV        : 1;
  __REG32 SWDPW        : 1;
  __REG32              : 1;
  __REG32 SWONCE       : 1;
  __REG32 SWP120       : 1;
  __REG32              :22;
} __pmd_po_dsw_bits;

/*PMD CO_DSEL0 (CO0 PWM Toggle Register)*/
typedef struct {
  __REG32 SWFP0        : 1;
  __REG32              : 3;
  __REG32 SWCUX0       : 2;
  __REG32              :26;
} __pmd_co_dsel0_bits;

/*PMD CO_DSEL1 (CO1 PWM Toggle Register)*/
typedef struct {
  __REG32 SWFP1        : 1;
  __REG32              : 3;
  __REG32 SWCUX1       : 2;
  __REG32              :26;
} __pmd_co_dsel1_bits;

/*PMD ADTRGSEL (A/D Conversion Trigger Select Register)*/
typedef struct {
  __REG32 PTS0         : 5;
  __REG32 PTS1         : 5;
  __REG32 PTS2         : 5;
  __REG32              :17;
} __pmd_adtrgsel_bits;

/*PMD RATESW (CAR2 RATE Select Register)*/
typedef struct {
  __REG32 SWRC1        : 1;
  __REG32              :31;
} __pmd_ratesw_bits;

/*PMD PCSR0 (Free PWM0 Cycle Set Register)*/
typedef struct {
  __REG32 PCSR0        :16;
  __REG32              :16;
} __pmd_pcsr0_bits;

/*PMD PDOUT0 (Free PWM0 Pulse Width Set Register)*/
typedef struct {
  __REG32 PDOUT0       :16;
  __REG32              :16;
} __pmd_pdout0_bits;

/*PMD PWMC0 (Counter 0 Control Register)*/
typedef struct {
  __REG32 PWMCK        : 2;
  __REG32              : 3;
  __REG32 PWMACT       : 1;
  __REG32              :26;
} __pmd_pwmc0_bits;

/*PMD CNT0 (Counter 0 Set Register)*/
typedef struct {
  __REG32 CNT0         :16;
  __REG32              :16;
} __pmd_cnt0_bits;

/*PMD PCSR1 (Free PWM1 Cycle Set Register)*/
typedef struct {
  __REG32 PCSR1        :16;
  __REG32              :16;
} __pmd_pcsr1_bits;

/*PMD PDOUT1 (Free PWM1 Pulse Width Set Register)*/
typedef struct {
  __REG32 PDOUT1       :16;
  __REG32              :16;
} __pmd_pdout1_bits;

/*PMD PWMC1 (Counter 1 Control Register)*/
typedef struct {
  __REG32 PWMCK        : 2;
  __REG32              : 3;
  __REG32 PWMACT       : 1;
  __REG32              :26;
} __pmd_pwmc1_bits;

/*PMD PWMC1 (Counter 1 Control Register)*/
typedef struct {
  __REG32 CNT1         :16;
  __REG32              :16;
} __pmd_cnt1_bits;

/*PMD SPWMQA (Sync PWM Q [A] Set Register)*/
typedef struct {
  __REG32 SPWMQA       :14;
  __REG32              :18;
} __pmd_spwmqa_bits;

/*PMD SPWMQB (Sync PWM Q [B] Set Register)*/
typedef struct {
  __REG32 SPWMQB       :14;
  __REG32              :18;
} __pmd_spwmqb_bits;

/*PMD SPWMQC (Sync PWM Q [C] Set Register)*/
typedef struct {
  __REG32 SPWMQC       :14;
  __REG32              :18;
} __pmd_spwmqc_bits;

/*PMD SPWMQD (Sync PWM Q [D] Set Register)*/
typedef struct {
  __REG32 SPWMQD       :14;
  __REG32              :18;
} __pmd_spwmqd_bits;

/*PMD SPWMQE (Sync PWM Q [E] Set Register)*/
typedef struct {
  __REG32 SPWMQE       :14;
  __REG32              :18;
} __pmd_spwmqe_bits;

/*PMD SPWMQF (Sync PWM Q [F] Set Register)*/
typedef struct {
  __REG32 SPWMQF       :14;
  __REG32              :18;
} __pmd_spwmqf_bits;

/*PMD SPWMQG (Sync PWM Q [G] Set Register)*/
typedef struct {
  __REG32 SPWMQG       :14;
  __REG32              :18;
} __pmd_spwmqg_bits;

/*PMD SPWMQH (Sync PWM Q [H] Set Register)*/
typedef struct {
  __REG32 SPWMQH       :14;
  __REG32              :18;
} __pmd_spwmqh_bits;

/*PMD SPWMQI (Sync PWM Q [I] Set Register)*/
typedef struct {
  __REG32 SPWMQI       :14;
  __REG32              :18;
} __pmd_spwmqi_bits;

/*PMD SPWMQJ (Sync PWM Q [J] Set Register)*/
typedef struct {
  __REG32 SPWMQJ       :14;
  __REG32              :18;
} __pmd_spwmqj_bits;

/*PMD SPWMQK (Sync PWM Q [K] Set Register)*/
typedef struct {
  __REG32 SPWMQK       :14;
  __REG32              :18;
} __pmd_spwmqk_bits;

/*PMD SPWMQL (Sync PWM Q [L] Set Register)*/
typedef struct {
  __REG32 SPWMQL       :14;
  __REG32              :18;
} __pmd_spwmql_bits;

/*PMD SPWMQM (Sync PWM Q [M] Set Register)*/
typedef struct {
  __REG32 SPWMQM       :14;
  __REG32              :18;
} __pmd_spwmqm_bits;

/*PMD CO_DTR0 (CO-0 Dead Time Set Register)*/
/*PMD CO_DTR1 (CO-1 Dead Time Set Register)*/
typedef struct {
  __REG32 DTR          :12;
  __REG32              :20;
} __pmd_co_dtr0_bits;

/*PMD CO_MPR0 (CO-0 Minimum ON Period Set Register)*/
/*PMD CO_MPR1 (CO-1 Minimum ON Period Set Register)*/
typedef struct {
  __REG32 MPR          :12;
  __REG32              :20;
} __pmd_co_mpr0_bits;

/*PMD CO_MDEN0 (CO-0 Output Enable Register)*/
/*PMD CO_MDEN1 (CO-1 Output Enable Register)*/
typedef struct {
  __REG32 PWMEN        : 1;
  __REG32              :31;
} __pmd_co_mden0_bits;

/*PMD CO_PORTMD0 (CO-0 Port Output Mode Register)*/
/*PMD CO_PORTMD1 (CO-1 Port Output Mode Register)*/
typedef struct {
  __REG32 PORTMD       : 2;
  __REG32              :30;
} __pmd_co_portmd0_bits;

/*PMD CO_MDCR0 (CO-0 Output Control Register)*/
/*PMD CO_MDCR1 (CO-1 Output Control Register)*/
typedef struct {
  __REG32              : 5;
  __REG32 SYNTMD       : 1;
  __REG32              :26;
} __pmd_co_mdcr0_bits;

/*PMD CO_MDOUT0 (CO-0 Output Waveform Control Register)*/
/*PMD CO_MDOUT1 (CO-1 Output Waveform Control Register)*/
typedef struct {
  __REG32 COC          : 2;
  __REG32              : 6;
  __REG32 CPWM         : 1;
  __REG32              :23;
} __pmd_co_mdout0_bits;

/*PMD CO_MDPOT0 (CO-0 Output Set Register)*/
/*PMD CO_MDPOT1 (CO-1 Output Set Register)*/
typedef struct {
  __REG32 PSYNCS       : 2;
  __REG32 POLL         : 1;
  __REG32 POLH         : 1;
  __REG32              :28;
} __pmd_co_mdpot0_bits;

/*PMD CO_EMGREL0 (CO-0 EMG Release Register)*/
/*PMD CO_EMGREL1 (CO-1 EMG Release Register)*/
typedef struct {
  __REG32 EMGREL       : 8;
  __REG32              :24;
} __pmd_co_emgrel0_bits;

/*PMD CO_EMGCR0 (CO-0 EMG Control Register)*/
/*PMD CO_EMGCR1 (CO-1 EMG Control Register)*/
typedef struct {
  __REG32 EMGEN        : 1;
  __REG32 EMGRS        : 1;
  __REG32              : 1;
  __REG32 EMGMD        : 2;
  __REG32 INHEN        : 1;
  __REG32              : 2;
  __REG32 EMGCNT       : 4;
  __REG32              :20;
} __pmd_co_emgcr0_bits;

/*PMD CO_EMGSTA0 (CO-0 Status Register)*/
typedef struct {
  __REG32 EMGST        : 1;
  __REG32 EMGI         : 1;
  __REG32 COU0_MP      : 1;
  __REG32 COL0_MP      : 1;
  __REG32 COU0_OUT     : 1;
  __REG32 COL0_OUT     : 1;
  __REG32              :26;
} __pmd_co_emgsta0_bits;

/*PMD CO_EMGSTA1 (CO-1 EMG Status Register)*/
typedef struct {
  __REG32 EMGST        : 1;
  __REG32 EMGI         : 1;
  __REG32 COU1_MP      : 1;
  __REG32 COL1_MP      : 1;
  __REG32 COU1_OUT     : 1;
  __REG32 COL1_OUT     : 1;
  __REG32              :26;
} __pmd_co_emgsta1_bits;

/*PMD NCMP (120° Power Update Timing Set Register)*/
typedef struct {
  __REG32 NCMP         :16;
  __REG32              :16;
} __pmd_ncmp_bits;

/*PMD SCMP (120° Power Instruction Value Register)*/
typedef struct {
  __REG32 SCMP         :14;
  __REG32              :18;
} __pmd_scmp_bits;

/*PMD P120CR (120° Power Control Register)*/
typedef struct {
  __REG32 SWPUVW       : 1;
  __REG32 P120TS       : 1;
  __REG32              :30;
} __pmd_p120cr_bits;

/*RSLTn (A/D Conversion Result Register n, n = 0 to 13)*/
typedef struct {
  __REG32  OVWR     : 1;
  __REG32  UVWXYZ   : 6;
  __REG32           : 5;
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

/*SECBIT (Security Set Register)*/
typedef struct {
  __REG32  SECBIT   : 1;
  __REG32           :31;
} __secbit_bits;

/*FLCSR0 (Flash Memory Control Status Register 0)*/
typedef struct {
  __REG32 RDY_BSY : 1;
  __REG32         :15;
  __REG32 BP0     : 1;
  __REG32 BP1     : 1;
  __REG32 BP2     : 1;
  __REG32 BP3     : 1;
  __REG32 BP4     : 1;
  __REG32 BP5     : 1;
  __REG32 BP6     : 1;
  __REG32 BP7     : 1;
  __REG32 BP8     : 1;
  __REG32 BP9     : 1;
  __REG32         : 6;
} __flcsr0_bits;

/*FLCSR1 (Flash Memory Control Status Register 1)*/
typedef struct {
  __REG32 RDY_BSY : 1;
  __REG32         :15;
  __REG32 BP10    : 1;
  __REG32 BP11    : 1;
  __REG32 BP12    : 1;
  __REG32 BP13    : 1;
  __REG32 BP14    : 1;
  __REG32 BP15    : 1;
  __REG32 BP16    : 1;
  __REG32 BP17    : 1;
  __REG32 BP18    : 1;
  __REG32 BP19    : 1;
  __REG32         : 6;
} __flcsr1_bits;

/*FLSR0 (Flash Memory Status Register 0)*/
typedef struct {
  __REG32 RDFA    : 1;
  __REG32 TRMA    : 1;
  __REG32 OPTA    : 1;
  __REG32 UTRA    : 1;
  __REG32         :28;
} __flsr0_bits;

/*FLSR1 (Flash Memory Status Register 1)*/
typedef struct {
  __REG32 UFLG    : 6;
  __REG32 ASWP    : 2;
  __REG32         :24;
} __flsr1_bits;

/*FLCR0 (Flash Memory Control Register 0)*/
typedef struct {
  __REG32 HRST    : 3;
  __REG32         :29;
} __flcr0_bits;

/*OVLADRn (Overlay Address Register n (n=0 through 3)*/
typedef struct {
  __REG32         :11;
  __REG32 OVLSTA  : 9;
  __REG32         :12;
} __ovladr_bits;

/*OVLEN (Overlay Enable Register)*/
typedef struct {
  __REG32 OVLEN0  : 1;
  __REG32 OVLEN1  : 1;
  __REG32 OVLEN2  : 1;
  __REG32 OVLEN3  : 1;
  __REG32         :28;
} __ovlen_bits;

/*OVLMOD (Overlay Mode Register)*/
typedef struct {
  __REG32 OVLMOD  : 8;
  __REG32         :24;
} __ovlmod_bits;

/*fRDATA (fRNET Read Data Register)*/
typedef struct {
  __REG32 RDATA   :24;
  __REG32         : 6;
  __REG32 STAT    : 2;
} __frdata_bits;

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

/* Interrupt Set-Enable Registers 64-95 */
typedef struct {
  __REG32  SETENA64       : 1;
  __REG32  SETENA65       : 1;
  __REG32  SETENA66       : 1;
  __REG32  SETENA67       : 1;
  __REG32  SETENA68       : 1;
  __REG32  SETENA69       : 1;
  __REG32  SETENA70       : 1;
  __REG32  SETENA71       : 1;
  __REG32  SETENA72       : 1;
  __REG32  SETENA73       : 1;
  __REG32  SETENA74       : 1;
  __REG32  SETENA75       : 1;
  __REG32  SETENA76       : 1;
  __REG32  SETENA77       : 1;
  __REG32  SETENA78       : 1;
  __REG32  SETENA79       : 1;
  __REG32  SETENA80       : 1;
  __REG32  SETENA81       : 1;
  __REG32  SETENA82       : 1;
  __REG32  SETENA83       : 1;
  __REG32  SETENA84       : 1;
  __REG32  SETENA85       : 1;
  __REG32  SETENA86       : 1;
  __REG32  SETENA87       : 1;
  __REG32  SETENA88       : 1;
  __REG32  SETENA89       : 1;
  __REG32  SETENA90       : 1;
  __REG32  SETENA91       : 1;
  __REG32  SETENA92       : 1;
  __REG32  SETENA93       : 1;
  __REG32  SETENA94       : 1;
  __REG32  SETENA95       : 1;
} __setena2_bits;

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

/* Interrupt Clear-Enable Registers 64-95 */
typedef struct {
  __REG32  CLRENA64       : 1;
  __REG32  CLRENA65       : 1;
  __REG32  CLRENA66       : 1;
  __REG32  CLRENA67       : 1;
  __REG32  CLRENA68       : 1;
  __REG32  CLRENA69       : 1;
  __REG32  CLRENA70       : 1;
  __REG32  CLRENA71       : 1;
  __REG32  CLRENA72       : 1;
  __REG32  CLRENA73       : 1;
  __REG32  CLRENA74       : 1;
  __REG32  CLRENA75       : 1;
  __REG32  CLRENA76       : 1;
  __REG32  CLRENA77       : 1;
  __REG32  CLRENA78       : 1;
  __REG32  CLRENA79       : 1;
  __REG32  CLRENA80       : 1;
  __REG32  CLRENA81       : 1;
  __REG32  CLRENA82       : 1;
  __REG32  CLRENA83       : 1;
  __REG32  CLRENA84       : 1;
  __REG32  CLRENA85       : 1;
  __REG32  CLRENA86       : 1;
  __REG32  CLRENA87       : 1;
  __REG32  CLRENA88       : 1;
  __REG32  CLRENA89       : 1;
  __REG32  CLRENA90       : 1;
  __REG32  CLRENA91       : 1;
  __REG32  CLRENA92       : 1;
  __REG32  CLRENA93       : 1;
  __REG32  CLRENA94       : 1;
  __REG32  CLRENA95       : 1;
} __clrena2_bits;

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

/* Interrupt Set-Pending Register 64-95 */
typedef struct {
  __REG32  SETPEND64      : 1;
  __REG32  SETPEND65      : 1;
  __REG32  SETPEND66      : 1;
  __REG32  SETPEND67      : 1;
  __REG32  SETPEND68      : 1;
  __REG32  SETPEND69      : 1;
  __REG32  SETPEND70      : 1;
  __REG32  SETPEND71      : 1;
  __REG32  SETPEND72      : 1;
  __REG32  SETPEND73      : 1;
  __REG32  SETPEND74      : 1;
  __REG32  SETPEND75      : 1;
  __REG32  SETPEND76      : 1;
  __REG32  SETPEND77      : 1;
  __REG32  SETPEND78      : 1;
  __REG32  SETPEND79      : 1;
  __REG32  SETPEND80      : 1;
  __REG32  SETPEND81      : 1;
  __REG32  SETPEND82      : 1;
  __REG32  SETPEND83      : 1;
  __REG32  SETPEND84      : 1;
  __REG32  SETPEND85      : 1;
  __REG32  SETPEND86      : 1;
  __REG32  SETPEND87      : 1;
  __REG32  SETPEND88      : 1;
  __REG32  SETPEND89      : 1;
  __REG32  SETPEND90      : 1;
  __REG32  SETPEND91      : 1;
  __REG32  SETPEND92      : 1;
  __REG32  SETPEND93      : 1;
  __REG32  SETPEND94      : 1;
  __REG32  SETPEND95      : 1;
} __setpend2_bits;

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

/* Interrupt Clear-Pending Register 64-95 */
typedef struct {
  __REG32  CLRPEND64      : 1;
  __REG32  CLRPEND65      : 1;
  __REG32  CLRPEND66      : 1;
  __REG32  CLRPEND67      : 1;
  __REG32  CLRPEND68      : 1;
  __REG32  CLRPEND69      : 1;
  __REG32  CLRPEND70      : 1;
  __REG32  CLRPEND71      : 1;
  __REG32  CLRPEND72      : 1;
  __REG32  CLRPEND73      : 1;
  __REG32  CLRPEND74      : 1;
  __REG32  CLRPEND75      : 1;
  __REG32  CLRPEND76      : 1;
  __REG32  CLRPEND77      : 1;
  __REG32  CLRPEND78      : 1;
  __REG32  CLRPEND79      : 1;
  __REG32  CLRPEND80      : 1;
  __REG32  CLRPEND81      : 1;
  __REG32  CLRPEND82      : 1;
  __REG32  CLRPEND83      : 1;
  __REG32  CLRPEND84      : 1;
  __REG32  CLRPEND85      : 1;
  __REG32  CLRPEND86      : 1;
  __REG32  CLRPEND87      : 1;
  __REG32  CLRPEND88      : 1;
  __REG32  CLRPEND89      : 1;
  __REG32  CLRPEND90      : 1;
  __REG32  CLRPEND91      : 1;
  __REG32  CLRPEND92      : 1;
  __REG32  CLRPEND93      : 1;
  __REG32  CLRPEND94      : 1;
  __REG32  CLRPEND95      : 1;
} __clrpend2_bits;

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

/* Active Bit Register 64-95 */
typedef struct {
  __REG32  ACTIVE64       : 1;
  __REG32  ACTIVE65       : 1;
  __REG32  ACTIVE66       : 1;
  __REG32  ACTIVE67       : 1;
  __REG32  ACTIVE68       : 1;
  __REG32  ACTIVE69       : 1;
  __REG32  ACTIVE70       : 1;
  __REG32  ACTIVE71       : 1;
  __REG32  ACTIVE72       : 1;
  __REG32  ACTIVE73       : 1;
  __REG32  ACTIVE74       : 1;
  __REG32  ACTIVE75       : 1;
  __REG32  ACTIVE76       : 1;
  __REG32  ACTIVE77       : 1;
  __REG32  ACTIVE78       : 1;
  __REG32  ACTIVE79       : 1;
  __REG32  ACTIVE80       : 1;
  __REG32  ACTIVE81       : 1;
  __REG32  ACTIVE82       : 1;
  __REG32  ACTIVE83       : 1;
  __REG32  ACTIVE84       : 1;
  __REG32  ACTIVE85       : 1;
  __REG32  ACTIVE86       : 1;
  __REG32  ACTIVE87       : 1;
  __REG32  ACTIVE88       : 1;
  __REG32  ACTIVE89       : 1;
  __REG32  ACTIVE90       : 1;
  __REG32  ACTIVE91       : 1;
  __REG32  ACTIVE92       : 1;
  __REG32  ACTIVE93       : 1;
  __REG32  ACTIVE94       : 1;
  __REG32  ACTIVE95       : 1;
} __active2_bits;

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

/* Interrupt Priority Registers 64-67 */
typedef struct {
  __REG32  PRI_64         : 8;
  __REG32  PRI_65         : 8;
  __REG32  PRI_66         : 8;
  __REG32  PRI_67         : 8;
} __pri16_bits;

/* Interrupt Priority Registers 68-71 */
typedef struct {
  __REG32  PRI_68         : 8;
  __REG32  PRI_69         : 8;
  __REG32  PRI_70         : 8;
  __REG32  PRI_71         : 8;
} __pri17_bits;

/* Interrupt Priority Registers 72-75 */
typedef struct {
  __REG32  PRI_72         : 8;
  __REG32  PRI_73         : 8;
  __REG32  PRI_74         : 8;
  __REG32  PRI_75         : 8;
} __pri18_bits;

/* Interrupt Priority Registers 76-79 */
typedef struct {
  __REG32  PRI_76         : 8;
  __REG32  PRI_77         : 8;
  __REG32  PRI_78         : 8;
  __REG32  PRI_79         : 8;
} __pri19_bits;

/* Interrupt Priority Registers 80-83 */
typedef struct {
  __REG32  PRI_80         : 8;
  __REG32  PRI_81         : 8;
  __REG32  PRI_82         : 8;
  __REG32  PRI_83         : 8;
} __pri20_bits;

/* Interrupt Priority Registers 84-87 */
typedef struct {
  __REG32  PRI_84         : 8;
  __REG32  PRI_85         : 8;
  __REG32  PRI_86         : 8;
  __REG32  PRI_87         : 8;
} __pri21_bits;

/* Interrupt Priority Registers 88-91 */
typedef struct {
  __REG32  PRI_88         : 8;
  __REG32  PRI_89         : 8;
  __REG32  PRI_90         : 8;
  __REG32  PRI_91         : 8;
} __pri22_bits;

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
__IO_REG32_BIT(SETENA2,             0xE000E108,__READ_WRITE ,__setena2_bits);
__IO_REG32_BIT(CLRENA0,             0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(CLRENA1,             0xE000E184,__READ_WRITE ,__clrena1_bits);
__IO_REG32_BIT(CLRENA2,             0xE000E188,__READ_WRITE ,__clrena2_bits);
__IO_REG32_BIT(SETPEND0,            0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(SETPEND1,            0xE000E204,__READ_WRITE ,__setpend1_bits);
__IO_REG32_BIT(SETPEND2,            0xE000E208,__READ_WRITE ,__setpend2_bits);
__IO_REG32_BIT(CLRPEND0,            0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(CLRPEND1,            0xE000E284,__READ_WRITE ,__clrpend1_bits);
__IO_REG32_BIT(CLRPEND2,            0xE000E288,__READ_WRITE ,__clrpend2_bits);
__IO_REG32_BIT(ACTIVE0,             0xE000E300,__READ       ,__active0_bits);
__IO_REG32_BIT(ACTIVE1,             0xE000E304,__READ       ,__active1_bits);
__IO_REG32_BIT(ACTIVE2,             0xE000E308,__READ       ,__active2_bits);
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
__IO_REG32_BIT(IP16,                0xE000E440,__READ_WRITE ,__pri16_bits);
__IO_REG32_BIT(IP17,                0xE000E444,__READ_WRITE ,__pri17_bits);
__IO_REG32_BIT(IP18,                0xE000E448,__READ_WRITE ,__pri18_bits);
__IO_REG32_BIT(IP19,                0xE000E44C,__READ_WRITE ,__pri19_bits);
__IO_REG32_BIT(IP20,                0xE000E450,__READ_WRITE ,__pri20_bits);
__IO_REG32_BIT(IP21,                0xE000E454,__READ_WRITE ,__pri21_bits);
__IO_REG32_BIT(IP22,                0xE000E458,__READ_WRITE ,__pri22_bits);
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
 ** MISC
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
__IO_REG32_BIT(FNCSELK,             0x40020020,__READ_WRITE ,__fncselk_bits);
__IO_REG32_BIT(PDENA,             	0x40020100,__READ_WRITE ,__pdena_bits);
__IO_REG32_BIT(PDENB,             	0x40020104,__READ_WRITE ,__pdenb_bits);
__IO_REG32_BIT(PDENC,             	0x40020108,__READ_WRITE ,__pdenc_bits);
__IO_REG32_BIT(PDEND,             	0x4002010C,__READ_WRITE ,__pdend_bits);
__IO_REG32_BIT(PDENE,             	0x40020110,__READ_WRITE ,__pdene_bits);
__IO_REG32_BIT(PDENF,             	0x40020114,__READ_WRITE ,__pdenf_bits);
__IO_REG32_BIT(PDENG,             	0x40020118,__READ_WRITE ,__pdeng_bits);
__IO_REG32_BIT(PDENH,             	0x4002011C,__READ_WRITE ,__pdenh_bits);
__IO_REG32_BIT(PDENK,             	0x40020120,__READ_WRITE ,__pdenk_bits);
__IO_REG32_BIT(PUENA,             	0x40020180,__READ_WRITE ,__puena_bits);
__IO_REG32_BIT(PUENB,             	0x40020184,__READ_WRITE ,__puenb_bits);
__IO_REG32_BIT(PUENC,             	0x40020188,__READ_WRITE ,__puenc_bits);
__IO_REG32_BIT(PUEND,             	0x4002018C,__READ_WRITE ,__puend_bits);
__IO_REG32_BIT(PUENE,             	0x40020190,__READ_WRITE ,__puene_bits);
__IO_REG32_BIT(PUENF,             	0x40020194,__READ_WRITE ,__puenf_bits);
__IO_REG32_BIT(PUENG,             	0x40020198,__READ_WRITE ,__pueng_bits);
__IO_REG32_BIT(PUENH,             	0x4002019C,__READ_WRITE ,__puenh_bits);
__IO_REG32_BIT(PUENK,             	0x400201A0,__READ_WRITE ,__puenk_bits);
__IO_REG32_BIT(SSOCTL,              0x40020200,__READ_WRITE ,__ssoctl_bits);
__IO_REG32_BIT(CANM,             		0x40020204,__READ_WRITE ,__canm_bits);
__IO_REG32_BIT(SIOCKEN,             0x40020208,__READ_WRITE ,__siocken_bits);
__IO_REG32_BIT(ADCCTL,              0x4002020C,__READ_WRITE ,__adcctl_bits);
__IO_REG32_BIT(CAL_TRIG_EN,         0x40020F00,__READ_WRITE ,__cal_trig_en_bits);
__IO_REG32(    CAL_TRIG_SET,        0x40020F04,__READ_WRITE );
__IO_REG32(    CAL_TRIG_CLR,        0x40020F08,__READ       );
__IO_REG32(    CAL_STARTUP,         0x40020F0C,__READ_WRITE );

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(MONA,           			0x40021000,__READ       ,__mona_bits);
__IO_REG32_BIT(OUTA,           			0x40021004,__READ_WRITE ,__outa_bits);
__IO_REG32_BIT(OENA,           			0x40021008,__READ_WRITE ,__oena_bits);
__IO_REG32_BIT(MONB,           			0x40021100,__READ       ,__monb_bits);
__IO_REG32_BIT(OUTB,           			0x40021104,__READ_WRITE ,__outb_bits);
__IO_REG32_BIT(OENB,           			0x40021108,__READ_WRITE ,__oenb_bits);
__IO_REG32_BIT(MONC,           			0x40021200,__READ       ,__monc_bits);
__IO_REG32_BIT(OUTC,           			0x40021204,__READ_WRITE ,__outc_bits);
__IO_REG32_BIT(OENC,           			0x40021208,__READ_WRITE ,__oenc_bits);
__IO_REG32_BIT(MOND,           			0x40021300,__READ       ,__mond_bits);
__IO_REG32_BIT(OUTD,           			0x40021304,__READ_WRITE ,__outd_bits);
__IO_REG32_BIT(OEND,           			0x40021308,__READ_WRITE ,__oend_bits);
__IO_REG32_BIT(MONE,           			0x40021400,__READ       ,__mone_bits);
__IO_REG32_BIT(OUTE,           			0x40021404,__READ_WRITE ,__oute_bits);
__IO_REG32_BIT(OENE,           			0x40021408,__READ_WRITE ,__oene_bits);
__IO_REG32_BIT(MONF,           			0x40021500,__READ       ,__monf_bits);
__IO_REG32_BIT(OUTF,           			0x40021504,__READ_WRITE ,__outf_bits);
__IO_REG32_BIT(OENF,           			0x40021508,__READ_WRITE ,__oenf_bits);
__IO_REG32_BIT(MONG,           			0x40021600,__READ       ,__mong_bits);
__IO_REG32_BIT(OUTG,           			0x40021604,__READ_WRITE ,__outg_bits);
__IO_REG32_BIT(OENG,           			0x40021608,__READ_WRITE ,__oeng_bits);
__IO_REG32_BIT(MONH,           			0x40021700,__READ       ,__monh_bits);
__IO_REG32_BIT(OUTH,           			0x40021704,__READ_WRITE ,__outh_bits);
__IO_REG32_BIT(OENH,           			0x40021708,__READ_WRITE ,__oenh_bits);
__IO_REG32_BIT(MONK,           			0x40021800,__READ       ,__monk_bits);
__IO_REG32_BIT(OUTK,           			0x40021804,__READ_WRITE ,__outk_bits);
__IO_REG32_BIT(OENK,           			0x40021808,__READ_WRITE ,__oenk_bits);
__IO_REG32_BIT(MONM,           			0x40021900,__READ       ,__monm_bits);

/***************************************************************************
 **
 ** DMAC0
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA0CEN,              0xE0042004,__READ_WRITE ,__dmacen_bits);
__IO_REG32_BIT(DMA0REQ,              0xE0042008,__READ_WRITE ,__dmareq_bits);
__IO_REG32_BIT(DMA0SUS,              0xE004200C,__READ_WRITE ,__dmasus_bits);
__IO_REG32_BIT(DMA0ACT,              0xE0042010,__READ_WRITE ,__dmaact_bits);
__IO_REG32_BIT(DMA0END,              0xE0042014,__READ_WRITE ,__dmaend_bits);
__IO_REG32_BIT(DMA0PRI,              0xE0042018,__READ_WRITE ,__dmapri_bits);
__IO_REG32_BIT(DMA0ENE,              0xE004201C,__READ_WRITE ,__dmaene_bits);
__IO_REG32(    DMA0DTAB,             0xE0042020,__READ_WRITE );
__IO_REG32(    DMA0EVAD,             0xE0042024,__READ_WRITE );
__IO_REG32_BIT(DMA0CHN,              0xE0042028,__READ       ,__dmachn_bits);
__IO_REG32_BIT(DMA0XFTYP,            0xE004202C,__READ       ,__dmaxftyp_bits);
__IO_REG32(    DMA0XFSAD,            0xE0042030,__READ       );
__IO_REG32(    DMA0XFDAD,            0xE0042034,__READ       );
__IO_REG32_BIT(DMA0XFSIZ,            0xE0042038,__READ       ,__dmaxfsiz_bits);
__IO_REG32(    DMA0DSADS,            0xE004203C,__READ       );
__IO_REG32_BIT(DMA0DSNUM,            0xE0042040,__READ       ,__dmadsnum_bits);
__IO_REG32_BIT(DMA0LRQ,              0xE0042044,__READ_WRITE ,__dmalrq_bits);
__IO_REG32_BIT(DMA0MSK,              0xE0042800,__READ_WRITE ,__dmamsk_bits);

/***************************************************************************
 **
 ** DMAC1
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA1CEN,              0xE0044004,__READ_WRITE ,__dmacen_bits);
__IO_REG32_BIT(DMA1REQ,              0xE0044008,__READ_WRITE ,__dmareq_bits);
__IO_REG32_BIT(DMA1SUS,              0xE004400C,__READ_WRITE ,__dmasus_bits);
__IO_REG32_BIT(DMA1ACT,              0xE0044010,__READ_WRITE ,__dmaact_bits);
__IO_REG32_BIT(DMA1END,              0xE0044014,__READ_WRITE ,__dmaend_bits);
__IO_REG32_BIT(DMA1PRI,              0xE0044018,__READ_WRITE ,__dmapri_bits);
__IO_REG32_BIT(DMA1ENE,              0xE004401C,__READ_WRITE ,__dmaene_bits);
__IO_REG32(    DMA1DTAB,             0xE0044020,__READ_WRITE );
__IO_REG32(    DMA1EVAD,             0xE0044024,__READ_WRITE );
__IO_REG32_BIT(DMA1CHN,              0xE0044028,__READ       ,__dmachn_bits);
__IO_REG32_BIT(DMA1XFTYP,            0xE004402C,__READ       ,__dmaxftyp_bits);
__IO_REG32(    DMA1XFSAD,            0xE0044030,__READ       );
__IO_REG32(    DMA1XFDAD,            0xE0044034,__READ       );
__IO_REG32_BIT(DMA1XFSIZ,            0xE0044038,__READ       ,__dmaxfsiz_bits);
__IO_REG32(    DMA1DSADS,            0xE004403C,__READ       );
__IO_REG32_BIT(DMA1DSNUM,            0xE0044040,__READ       ,__dmadsnum_bits);
__IO_REG32_BIT(DMA1LRQ,              0xE0044044,__READ_WRITE ,__dmalrq_bits);
__IO_REG32_BIT(DMA1MSK,              0xE0044800,__READ_WRITE ,__dmamsk_bits);

/***************************************************************************
 **
 ** TCMP0
 **
 ***************************************************************************/
__IO_REG32_BIT(TCMP0_TBTRUN,         0x40023000,__READ_WRITE ,__tcmp_tbtrun_bits);
__IO_REG32_BIT(TCMP0_TBTCR,          0x40023004,__READ_WRITE ,__tcmp_tbtcr_bits);
__IO_REG32(    TCMP0_TBTCNT,         0x40023008,__READ_WRITE );
__IO_REG32_BIT(TCMP0_CMP0CR,         0x40023800,__READ_WRITE ,__tcmp_cmpcr_bits);
__IO_REG32_BIT(TCMP0_CMP1CR,         0x40023804,__READ_WRITE ,__tcmp_cmpcr_bits);
__IO_REG32(    TCMP0_CMP0,           0x40023808,__READ_WRITE );
__IO_REG32(    TCMP0_CMP1,           0x4002380C,__READ_WRITE );
__IO_REG32_BIT(TCMP0_CMPDO0,         0x40023810,__READ_WRITE ,__tcmp_cmpdo_bits);
__IO_REG32_BIT(TCMP0_CMPDO1,         0x40023814,__READ_WRITE ,__tcmp_cmpdo_bits);
__IO_REG32_BIT(TCMP0_CMPSD0,         0x40023818,__READ       ,__tcmp_cmpsd_bits);
__IO_REG32_BIT(TCMP0_CMPSD1,         0x4002381C,__READ       ,__tcmp_cmpsd_bits);
__IO_REG32_BIT(TCMP0_CMPOM0,         0x40023820,__READ_WRITE ,__tcmp_cmpom_bits);
__IO_REG32_BIT(TCMP0_CMPOM1,         0x40023824,__READ_WRITE ,__tcmp_cmpom_bits);

/***************************************************************************
 **
 ** TCMP1
 **
 ***************************************************************************/
__IO_REG32_BIT(TCMP1_TBTRUN,         0x40024000,__READ_WRITE ,__tcmp_tbtrun_bits);
__IO_REG32_BIT(TCMP1_TBTCR,          0x40024004,__READ_WRITE ,__tcmp_tbtcr_bits);
__IO_REG32(    TCMP1_TBTCNT,         0x40024008,__READ_WRITE );
__IO_REG32_BIT(TCMP1_CMP0CR,         0x40024800,__READ_WRITE ,__tcmp_cmpcr_bits);
__IO_REG32_BIT(TCMP1_CMP1CR,         0x40024804,__READ_WRITE ,__tcmp_cmpcr_bits);
__IO_REG32(    TCMP1_CMP0,           0x40024808,__READ_WRITE );
__IO_REG32(    TCMP1_CMP1,           0x4002480C,__READ_WRITE );
__IO_REG32_BIT(TCMP1_CMPDO0,         0x40024810,__READ_WRITE ,__tcmp_cmpdo_bits);
__IO_REG32_BIT(TCMP1_CMPDO1,         0x40024814,__READ_WRITE ,__tcmp_cmpdo_bits);
__IO_REG32_BIT(TCMP1_CMPSD0,         0x40024818,__READ       ,__tcmp_cmpsd_bits);
__IO_REG32_BIT(TCMP1_CMPSD1,         0x4002481C,__READ       ,__tcmp_cmpsd_bits);
__IO_REG32_BIT(TCMP1_CMPOM0,         0x40024820,__READ_WRITE ,__tcmp_cmpom_bits);
__IO_REG32_BIT(TCMP1_CMPOM1,         0x40024824,__READ_WRITE ,__tcmp_cmpom_bits);

/***************************************************************************
 **
 ** TCMP2
 **
 ***************************************************************************/
__IO_REG32_BIT(TCMP2_TBTRUN,         0x40025000,__READ_WRITE ,__tcmp_tbtrun_bits);
__IO_REG32_BIT(TCMP2_TBTCR,          0x40025004,__READ_WRITE ,__tcmp_tbtcr_bits);
__IO_REG32(    TCMP2_TBTCNT,         0x40025008,__READ_WRITE );
__IO_REG32_BIT(TCMP2_CMP0CR,         0x40025800,__READ_WRITE ,__tcmp_cmpcr_bits);
__IO_REG32_BIT(TCMP2_CMP1CR,         0x40025804,__READ_WRITE ,__tcmp_cmpcr_bits);
__IO_REG32(    TCMP2_CMP0,           0x40025808,__READ_WRITE );
__IO_REG32(    TCMP2_CMP1,           0x4002580C,__READ_WRITE );
__IO_REG32_BIT(TCMP2_CMPDO0,         0x40025810,__READ_WRITE ,__tcmp_cmpdo_bits);
__IO_REG32_BIT(TCMP2_CMPDO1,         0x40025814,__READ_WRITE ,__tcmp_cmpdo_bits);
__IO_REG32_BIT(TCMP2_CMPSD0,         0x40025818,__READ       ,__tcmp_cmpsd_bits);
__IO_REG32_BIT(TCMP2_CMPSD1,         0x4002581C,__READ       ,__tcmp_cmpsd_bits);
__IO_REG32_BIT(TCMP2_CMPOM0,         0x40025820,__READ_WRITE ,__tcmp_cmpom_bits);
__IO_REG32_BIT(TCMP2_CMPOM1,         0x40025824,__READ_WRITE ,__tcmp_cmpom_bits);

/***************************************************************************
 **
 ** TCMP3
 **
 ***************************************************************************/
__IO_REG32_BIT(TCMP3_TBTRUN,         0x40026000,__READ_WRITE ,__tcmp_tbtrun_bits);
__IO_REG32_BIT(TCMP3_TBTCR,          0x40026004,__READ_WRITE ,__tcmp_tbtcr_bits);
__IO_REG32(    TCMP3_TBTCNT,         0x40026008,__READ_WRITE );
__IO_REG32_BIT(TCMP3_CMP0CR,         0x40026800,__READ_WRITE ,__tcmp_cmpcr_bits);
__IO_REG32_BIT(TCMP3_CMP1CR,         0x40026804,__READ_WRITE ,__tcmp_cmpcr_bits);
__IO_REG32(    TCMP3_CMP0,           0x40026808,__READ_WRITE );
__IO_REG32(    TCMP3_CMP1,           0x4002680C,__READ_WRITE );
__IO_REG32_BIT(TCMP3_CMPDO0,         0x40026810,__READ_WRITE ,__tcmp_cmpdo_bits);
__IO_REG32_BIT(TCMP3_CMPDO1,         0x40026814,__READ_WRITE ,__tcmp_cmpdo_bits);
__IO_REG32_BIT(TCMP3_CMPSD0,         0x40026818,__READ       ,__tcmp_cmpsd_bits);
__IO_REG32_BIT(TCMP3_CMPSD1,         0x4002681C,__READ       ,__tcmp_cmpsd_bits);
__IO_REG32_BIT(TCMP3_CMPOM0,         0x40026820,__READ_WRITE ,__tcmp_cmpom_bits);
__IO_REG32_BIT(TCMP3_CMPOM1,         0x40026824,__READ_WRITE ,__tcmp_cmpom_bits);

/***************************************************************************
 **
 ** TCMP4
 **
 ***************************************************************************/
__IO_REG32_BIT(TCMP4_TBTRUN,         0x40027000,__READ_WRITE ,__tcmp_tbtrun_bits);
__IO_REG32_BIT(TCMP4_TBTCR,          0x40027004,__READ_WRITE ,__tcmp_tbtcr_bits);
__IO_REG32(    TCMP4_TBTCNT,         0x40027008,__READ_WRITE );
__IO_REG32_BIT(TCMP4_CMP0CR,         0x40027800,__READ_WRITE ,__tcmp_cmpcr_bits);
__IO_REG32_BIT(TCMP4_CMP1CR,         0x40027804,__READ_WRITE ,__tcmp_cmpcr_bits);
__IO_REG32(    TCMP4_CMP0,           0x40027808,__READ_WRITE );
__IO_REG32(    TCMP4_CMP1,           0x4002780C,__READ_WRITE );
__IO_REG32_BIT(TCMP4_CMPDO0,         0x40027810,__READ_WRITE ,__tcmp_cmpdo_bits);
__IO_REG32_BIT(TCMP4_CMPDO1,         0x40027814,__READ_WRITE ,__tcmp_cmpdo_bits);
__IO_REG32_BIT(TCMP4_CMPSD0,         0x40027818,__READ       ,__tcmp_cmpsd_bits);
__IO_REG32_BIT(TCMP4_CMPSD1,         0x4002781C,__READ       ,__tcmp_cmpsd_bits);
__IO_REG32_BIT(TCMP4_CMPOM0,         0x40027820,__READ_WRITE ,__tcmp_cmpom_bits);
__IO_REG32_BIT(TCMP4_CMPOM1,         0x40027824,__READ_WRITE ,__tcmp_cmpom_bits);

/***************************************************************************
 **
 ** TCAP0
 **
 ***************************************************************************/
__IO_REG32_BIT(TCAP0_TBTRUN,          0x4002C000,__READ_WRITE ,__tcap_tbtrun_bits);
__IO_REG32_BIT(TCAP0_TBTCR,           0x4002C004,__READ_WRITE ,__tcap_tbtcr_bits);
__IO_REG32(    TCAP0_TBTCNT,          0x4002C008,__READ_WRITE );
__IO_REG32_BIT(TCAP0_INCAPCR,         0x4002C400,__READ_WRITE ,__tcap_incapcr_bits);
__IO_REG32(    TCAP0_INCAP0R,         0x4002C404,__READ_WRITE );
__IO_REG32(    TCAP0_INCAP0F,         0x4002C408,__READ       );
__IO_REG32(    TCAP0_INCAP1R,         0x4002C40C,__READ       );
__IO_REG32(    TCAP0_INCAP1F,         0x4002C410,__READ       );
__IO_REG32(    TCAP0_INCAP2R,         0x4002C414,__READ       );
__IO_REG32(    TCAP0_INCAP2F,         0x4002C418,__READ       );
__IO_REG32(    TCAP0_INCAP3R,         0x4002C41C,__READ       );
__IO_REG32(    TCAP0_INCAP3F,         0x4002C420,__READ       );
__IO_REG32(    TCAP0_INCAP4R,         0x4002C424,__READ       );
__IO_REG32(    TCAP0_INCAP4F,         0x4002C428,__READ       );
__IO_REG32(    TCAP0_INCAP5R,         0x4002C42C,__READ       );
__IO_REG32(    TCAP0_INCAP5F,         0x4002C430,__READ       );
__IO_REG32_BIT(TCAP0_CMP0CR,          0x4002C800,__READ_WRITE ,__tcap_cmpcr_bits);
__IO_REG32_BIT(TCAP0_CMP1CR,          0x4002C804,__READ_WRITE ,__tcap_cmpcr_bits);
__IO_REG32(    TCAP0_CMP0,            0x4002C808,__READ_WRITE );
__IO_REG32(    TCAP0_CMP1,            0x4002C80C,__READ_WRITE );

/***************************************************************************
 **
 ** TCAP1
 **
 ***************************************************************************/
__IO_REG32_BIT(TCAP1_TBTRUN,          0x4001D000,__READ_WRITE ,__tcap_tbtrun_bits);
__IO_REG32_BIT(TCAP1_TBTCR,           0x4001D004,__READ_WRITE ,__tcap_tbtcr_bits);
__IO_REG32(    TCAP1_TBTCNT,          0x4001D008,__READ_WRITE );
__IO_REG32(    TCAP1_INCAP0R,         0x4001D404,__READ_WRITE );
__IO_REG32(    TCAP1_INCAP1R,         0x4001D40C,__READ       );
__IO_REG32(    TCAP1_INCAP2R,         0x4001D414,__READ       );
__IO_REG32(    TCAP1_INCAP3R,         0x4001D41C,__READ       );
__IO_REG32(    TCAP1_INCAP3F,         0x4001D420,__READ       );
__IO_REG32(    TCAP1_INCAP4R,         0x4001D424,__READ       );
__IO_REG32(    TCAP1_INCAP5R,         0x4001D42C,__READ       );
__IO_REG32(    TCAP1_INCAP6R,         0x4001D434,__READ       );
__IO_REG32(    TCAP1_INCAP7R,         0x4001D43C,__READ       );
__IO_REG32(    TCAP1_INCAP7F,         0x4001D440,__READ       );

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWMSYNCRUN,          0x4002A000,__READ_WRITE ,__pwmsyncrun_bits);
__IO_REG32_BIT(PWM00_RUN,           0x4002A800,__READ_WRITE ,__pwm_run_bits);
__IO_REG32_BIT(PWM00_CTRL,          0x4002A804,__READ_WRITE ,__pwm_cr_bits);
__IO_REG32_BIT(PWM00_MOD,           0x4002A808,__READ_WRITE ,__pwm_mod_bits);
__IO_REG32_BIT(PWM00_OUTCTRL,       0x4002A80C,__READ_WRITE ,__pwm_outctrl_bits);
__IO_REG32_BIT(PWM00_PRICMP,        0x4002A810,__READ_WRITE ,__pwm_rg0_bits);
__IO_REG32_BIT(PWM00_DUTYCMP,       0x4002A814,__READ_WRITE ,__pwm_rg1_bits);
__IO_REG32_BIT(PWM00_CNT,           0x4002A818,__READ_WRITE ,__pwm_cnt_bits);
__IO_REG32_BIT(PWM01_RUN,           0x4002A880,__READ_WRITE ,__pwm_run_bits);
__IO_REG32_BIT(PWM01_CTRL,          0x4002A884,__READ_WRITE ,__pwm_cr_bits);
__IO_REG32_BIT(PWM01_MOD,           0x4002A888,__READ_WRITE ,__pwm_mod_bits);
__IO_REG32_BIT(PWM01_OUTCTRL,       0x4002A88C,__READ_WRITE ,__pwm_outctrl_bits);
__IO_REG32_BIT(PWM01_PRICMP,        0x4002A890,__READ_WRITE ,__pwm_rg0_bits);
__IO_REG32_BIT(PWM01_DUTYCMP,       0x4002A894,__READ_WRITE ,__pwm_rg1_bits);
__IO_REG32_BIT(PWM01_CNT,           0x4002A898,__READ_WRITE ,__pwm_cnt_bits);
__IO_REG32_BIT(PWM02_RUN,           0x4002A900,__READ_WRITE ,__pwm_run_bits);
__IO_REG32_BIT(PWM02_CTRL,          0x4002A904,__READ_WRITE ,__pwm_cr_bits);
__IO_REG32_BIT(PWM02_MOD,           0x4002A908,__READ_WRITE ,__pwm_mod_bits);
__IO_REG32_BIT(PWM02_OUTCTRL,       0x4002A90C,__READ_WRITE ,__pwm_outctrl_bits);
__IO_REG32_BIT(PWM02_PRICMP,        0x4002A910,__READ_WRITE ,__pwm_rg0_bits);
__IO_REG32_BIT(PWM02_DUTYCMP,       0x4002A914,__READ_WRITE ,__pwm_rg1_bits);
__IO_REG32_BIT(PWM02_CNT,           0x4002A918,__READ_WRITE ,__pwm_cnt_bits);
__IO_REG32_BIT(PWM03_RUN,           0x4002A980,__READ_WRITE ,__pwm_run_bits);
__IO_REG32_BIT(PWM03_CTRL,          0x4002A984,__READ_WRITE ,__pwm_cr_bits);
__IO_REG32_BIT(PWM03_MOD,           0x4002A988,__READ_WRITE ,__pwm_mod_bits);
__IO_REG32_BIT(PWM03_OUTCTRL,       0x4002A98C,__READ_WRITE ,__pwm_outctrl_bits);
__IO_REG32_BIT(PWM03_PRICMP,        0x4002A990,__READ_WRITE ,__pwm_rg0_bits);
__IO_REG32_BIT(PWM03_DUTYCMP,       0x4002A994,__READ_WRITE ,__pwm_rg1_bits);
__IO_REG32_BIT(PWM03_CNT,           0x4002A998,__READ_WRITE ,__pwm_cnt_bits);

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
 ** SIO2
 **
 ***************************************************************************/
__IO_REG8_BIT( SC2EN,             	0x4003F000,__READ_WRITE ,__scen_bits);
__IO_REG8(		 SC2BUF,             	0x4003F004,__READ_WRITE );
__IO_REG8_BIT( SC2CR,             	0x4003F008,__READ_WRITE ,__sccr_bits);
__IO_REG8_BIT( SC2MOD0,            	0x4003F00C,__READ_WRITE ,__scmod0_bits);
__IO_REG8_BIT( SC2BRCR,           	0x4003F010,__READ_WRITE ,__scbrcr_bits);
__IO_REG8_BIT( SC2BRADD,           	0x4003F014,__READ_WRITE ,__scbradd_bits);
__IO_REG8_BIT( SC2MOD1,           	0x4003F018,__READ_WRITE ,__scmod1_bits);
__IO_REG8_BIT( SC2MOD2,           	0x4003F01C,__READ_WRITE ,__scmod2_bits);
__IO_REG8_BIT( SC2RFC,             	0x4003F020,__READ_WRITE ,__scrfc_bits);
__IO_REG8_BIT( SC2TFC,             	0x4003F024,__READ_WRITE ,__sctfc_bits);
__IO_REG8_BIT( SC2RST,             	0x4003F028,__READ				,__scrst_bits);
__IO_REG8_BIT( SC2TST,             	0x4003F02C,__READ				,__sctst_bits);
__IO_REG8_BIT( SC2FCNF,            	0x4003F030,__READ_WRITE ,__scfcnf_bits);

/***************************************************************************
 **
 ** CAN0
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN0MB0ID,           0x40040000,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB0CR,           0x40040008,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB0DL,           0x40040010,__READ_WRITE );
__IO_REG32(    CAN0MB0DH,           0x40040018,__READ_WRITE );
__IO_REG32_BIT(CAN0MB1ID,           0x40040020,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB1CR,           0x40040028,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB1DL,           0x40040030,__READ_WRITE );
__IO_REG32(    CAN0MB1DH,           0x40040038,__READ_WRITE );
__IO_REG32_BIT(CAN0MB2ID,           0x40040040,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB2CR,           0x40040048,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB2DL,           0x40040050,__READ_WRITE );
__IO_REG32(    CAN0MB2DH,           0x40040058,__READ_WRITE );
__IO_REG32_BIT(CAN0MB3ID,           0x40040060,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB3CR,           0x40040068,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB3DL,           0x40040070,__READ_WRITE );
__IO_REG32(    CAN0MB3DH,           0x40040078,__READ_WRITE );
__IO_REG32_BIT(CAN0MB4ID,           0x40040080,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB4CR,           0x40040088,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB4DL,           0x40040090,__READ_WRITE );
__IO_REG32(    CAN0MB4DH,           0x40040098,__READ_WRITE );
__IO_REG32_BIT(CAN0MB5ID,           0x400400A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB5CR,           0x400400A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB5DL,           0x400400B0,__READ_WRITE );
__IO_REG32(    CAN0MB5DH,           0x400400B8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB6ID,           0x400400C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB6CR,           0x400400C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB6DL,           0x400400D0,__READ_WRITE );
__IO_REG32(    CAN0MB6DH,           0x400400D8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB7ID,           0x400400E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB7CR,           0x400400E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB7DL,           0x400400F0,__READ_WRITE );
__IO_REG32(    CAN0MB7DH,           0x400400F8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB8ID,           0x40040100,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB8CR,           0x40040108,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB8DL,           0x40040110,__READ_WRITE );
__IO_REG32(    CAN0MB8DH,           0x40040118,__READ_WRITE );
__IO_REG32_BIT(CAN0MB9ID,           0x40040120,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB9CR,           0x40040128,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB9DL,           0x40040130,__READ_WRITE );
__IO_REG32(    CAN0MB9DH,           0x40040138,__READ_WRITE );
__IO_REG32_BIT(CAN0MB10ID,          0x40040140,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB10CR,          0x40040148,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB10DL,          0x40040150,__READ_WRITE );
__IO_REG32(    CAN0MB10DH,          0x40040158,__READ_WRITE );
__IO_REG32_BIT(CAN0MB11ID,          0x40040160,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB11CR,          0x40040168,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB11DL,          0x40040170,__READ_WRITE );
__IO_REG32(    CAN0MB11DH,          0x40040178,__READ_WRITE );
__IO_REG32_BIT(CAN0MB12ID,          0x40040180,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB12CR,          0x40040188,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB12DL,          0x40040190,__READ_WRITE );
__IO_REG32(    CAN0MB12DH,          0x40040198,__READ_WRITE );
__IO_REG32_BIT(CAN0MB13ID,          0x400401A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB13CR,          0x400401A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB13DL,          0x400401B0,__READ_WRITE );
__IO_REG32(    CAN0MB13DH,          0x400401B8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB14ID,          0x400401C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB14CR,          0x400401C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB14DL,          0x400401D0,__READ_WRITE );
__IO_REG32(    CAN0MB14DH,          0x400401D8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB15ID,          0x400401E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB15CR,          0x400401E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB15DL,          0x400401F0,__READ_WRITE );
__IO_REG32(    CAN0MB15DH,          0x400401F8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB16ID,          0x40040200,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB16CR,          0x40040208,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB16DL,          0x40040210,__READ_WRITE );
__IO_REG32(    CAN0MB16DH,          0x40040218,__READ_WRITE );
__IO_REG32_BIT(CAN0MB17ID,          0x40040220,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB17CR,          0x40040228,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB17DL,          0x40040230,__READ_WRITE );
__IO_REG32(    CAN0MB17DH,          0x40040238,__READ_WRITE );
__IO_REG32_BIT(CAN0MB18ID,          0x40040240,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB18CR,          0x40040248,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB18DL,          0x40040250,__READ_WRITE );
__IO_REG32(    CAN0MB18DH,          0x40040258,__READ_WRITE );
__IO_REG32_BIT(CAN0MB19ID,          0x40040260,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB19CR,          0x40040268,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB19DL,          0x40040270,__READ_WRITE );
__IO_REG32(    CAN0MB19DH,          0x40040278,__READ_WRITE );
__IO_REG32_BIT(CAN0MB20ID,          0x40040280,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB20CR,          0x40040288,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB20DL,          0x40040290,__READ_WRITE );
__IO_REG32(    CAN0MB20DH,          0x40040298,__READ_WRITE );
__IO_REG32_BIT(CAN0MB21ID,          0x400402A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB21CR,          0x400402A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB21DL,          0x400402B0,__READ_WRITE );
__IO_REG32(    CAN0MB21DH,          0x400402B8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB22ID,          0x400402C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB22CR,          0x400402C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB22DL,          0x400402D0,__READ_WRITE );
__IO_REG32(    CAN0MB22DH,          0x400402D8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB23ID,          0x400402E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB23CR,          0x400402E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB23DL,          0x400402F0,__READ_WRITE );
__IO_REG32(    CAN0MB23DH,          0x400402F8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB24ID,          0x40040300,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB24CR,          0x40040308,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB24DL,          0x40040310,__READ_WRITE );
__IO_REG32(    CAN0MB24DH,          0x40040318,__READ_WRITE );
__IO_REG32_BIT(CAN0MB25ID,          0x40040320,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB25CR,          0x40040328,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB25DL,          0x40040330,__READ_WRITE );
__IO_REG32(    CAN0MB25DH,          0x40040338,__READ_WRITE );
__IO_REG32_BIT(CAN0MB26ID,          0x40040340,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB26CR,          0x40040348,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB26DL,          0x40040350,__READ_WRITE );
__IO_REG32(    CAN0MB26DH,          0x40040358,__READ_WRITE );
__IO_REG32_BIT(CAN0MB27ID,          0x40040360,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB27CR,          0x40040368,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB27DL,          0x40040370,__READ_WRITE );
__IO_REG32(    CAN0MB27DH,          0x40040378,__READ_WRITE );
__IO_REG32_BIT(CAN0MB28ID,          0x40040380,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB28CR,          0x40040388,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB28DL,          0x40040390,__READ_WRITE );
__IO_REG32(    CAN0MB28DH,          0x40040398,__READ_WRITE );
__IO_REG32_BIT(CAN0MB29ID,          0x400403A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB29CR,          0x400403A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB29DL,          0x400403B0,__READ_WRITE );
__IO_REG32(    CAN0MB29DH,          0x400403B8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB30ID,          0x400403C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB30CR,          0x400403C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB30DL,          0x400403D0,__READ_WRITE );
__IO_REG32(    CAN0MB30DH,          0x400403D8,__READ_WRITE );
__IO_REG32_BIT(CAN0MB31ID,          0x400403E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN0MB31CR,          0x400403E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN0MB31DL,          0x400403F0,__READ_WRITE );
__IO_REG32(    CAN0MB31DH,          0x400403F8,__READ_WRITE );
__IO_REG32_BIT(CAN0MC,              0x40040400,__READ_WRITE ,__canmc_bits);
__IO_REG32_BIT(CAN0MD,              0x40040408,__READ_WRITE ,__canmd_bits);
__IO_REG32_BIT(CAN0TRS,             0x40040410,__READ_WRITE ,__cantrs_bits);
__IO_REG32_BIT(CAN0TRR,             0x40040418,__READ_WRITE ,__cantrr_bits);
__IO_REG32_BIT(CAN0TA,              0x40040420,__READ_WRITE ,__canta_bits);
__IO_REG32_BIT(CAN0AA,              0x40040428,__READ_WRITE ,__canaa_bits);
__IO_REG32_BIT(CAN0RMP,             0x40040430,__READ_WRITE ,__canrmp_bits);
__IO_REG32_BIT(CAN0RML,             0x40040438,__READ_WRITE ,__canrml_bits);
__IO_REG32_BIT(CAN0MCR,             0x40040450,__READ_WRITE ,__canmcr_bits);
__IO_REG32_BIT(CAN0GSR,             0x40040458,__READ       ,__cangsr_bits);
__IO_REG32_BIT(CAN0BCR1,            0x40040460,__READ_WRITE ,__canbcr1_bits);
__IO_REG32_BIT(CAN0BCR2,            0x40040468,__READ_WRITE ,__canbcr2_bits);
__IO_REG32_BIT(CAN0GIF,             0x40040470,__READ_WRITE ,__cangif_bits);
__IO_REG32_BIT(CAN0GIM,             0x40040478,__READ_WRITE ,__cangim_bits);
__IO_REG32_BIT(CAN0MBTIF,           0x40040480,__READ_WRITE ,__canmbtif_bits);
__IO_REG32_BIT(CAN0MBRIF,           0x40040488,__READ_WRITE ,__canmbrif_bits);
__IO_REG32_BIT(CAN0MBIM,            0x40040490,__READ_WRITE ,__canmbim_bits);
__IO_REG32_BIT(CAN0CDR,             0x40040498,__READ_WRITE ,__cancdr_bits);
__IO_REG32_BIT(CAN0RFP,             0x400404A0,__READ_WRITE ,__canrfp_bits);
__IO_REG32_BIT(CAN0CEC,             0x400404A8,__READ       ,__cancec_bits);
__IO_REG32_BIT(CAN0TSP,             0x400404B0,__READ_WRITE ,__cantsp_bits);
__IO_REG32_BIT(CAN0TSC,             0x400404B8,__READ_WRITE ,__cantsc_bits);
__IO_REG32_BIT(CAN0MB0AM,           0x400404C0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB1AM,           0x400404C8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB2AM,           0x400404D0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB3AM,           0x400404D8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB4AM,           0x400404E0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB5AM,           0x400404E8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB6AM,           0x400404F0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB7AM,           0x400404F8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB8AM,           0x40040500,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB9AM,           0x40040508,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB10AM,          0x40040510,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB11AM,          0x40040518,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB12AM,          0x40040520,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB13AM,          0x40040528,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB14AM,          0x40040530,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB15AM,          0x40040538,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB16AM,          0x40040540,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB17AM,          0x40040548,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB18AM,          0x40040550,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB19AM,          0x40040558,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB20AM,          0x40040560,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB21AM,          0x40040568,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB22AM,          0x40040570,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB23AM,          0x40040578,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB24AM,          0x40040580,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB25AM,          0x40040588,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB26AM,          0x40040590,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB27AM,          0x40040598,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB28AM,          0x400405A0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB29AM,          0x400405A8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB30AM,          0x400405B0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN0MB31AM,          0x400405B8,__READ_WRITE ,__canmbam_bits);
__IO_REG32(		 CAN0INTCRCLR,        0x40040800,__READ_WRITE );
__IO_REG32(		 CAN0INTCTCLR,        0x40040808,__READ_WRITE );
__IO_REG32(		 CAN0INTCGCLR,        0x40040810,__READ_WRITE );

/***************************************************************************
 **
 ** CAN1
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN1MB0ID,           0x40041000,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB0CR,           0x40041008,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB0DL,           0x40041010,__READ_WRITE );
__IO_REG32(    CAN1MB0DH,           0x40041018,__READ_WRITE );
__IO_REG32_BIT(CAN1MB1ID,           0x40041020,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB1CR,           0x40041028,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB1DL,           0x40041030,__READ_WRITE );
__IO_REG32(    CAN1MB1DH,           0x40041038,__READ_WRITE );
__IO_REG32_BIT(CAN1MB2ID,           0x40041040,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB2CR,           0x40041048,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB2DL,           0x40041050,__READ_WRITE );
__IO_REG32(    CAN1MB2DH,           0x40041058,__READ_WRITE );
__IO_REG32_BIT(CAN1MB3ID,           0x40041060,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB3CR,           0x40041068,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB3DL,           0x40041070,__READ_WRITE );
__IO_REG32(    CAN1MB3DH,           0x40041078,__READ_WRITE );
__IO_REG32_BIT(CAN1MB4ID,           0x40041080,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB4CR,           0x40041088,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB4DL,           0x40041090,__READ_WRITE );
__IO_REG32(    CAN1MB4DH,           0x40041098,__READ_WRITE );
__IO_REG32_BIT(CAN1MB5ID,           0x400410A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB5CR,           0x400410A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB5DL,           0x400410B0,__READ_WRITE );
__IO_REG32(    CAN1MB5DH,           0x400410B8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB6ID,           0x400410C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB6CR,           0x400410C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB6DL,           0x400410D0,__READ_WRITE );
__IO_REG32(    CAN1MB6DH,           0x400410D8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB7ID,           0x400410E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB7CR,           0x400410E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB7DL,           0x400410F0,__READ_WRITE );
__IO_REG32(    CAN1MB7DH,           0x400410F8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB8ID,           0x40041100,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB8CR,           0x40041108,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB8DL,           0x40041110,__READ_WRITE );
__IO_REG32(    CAN1MB8DH,           0x40041118,__READ_WRITE );
__IO_REG32_BIT(CAN1MB9ID,           0x40041120,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB9CR,           0x40041128,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB9DL,           0x40041130,__READ_WRITE );
__IO_REG32(    CAN1MB9DH,           0x40041138,__READ_WRITE );
__IO_REG32_BIT(CAN1MB10ID,          0x40041140,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB10CR,          0x40041148,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB10DL,          0x40041150,__READ_WRITE );
__IO_REG32(    CAN1MB10DH,          0x40041158,__READ_WRITE );
__IO_REG32_BIT(CAN1MB11ID,          0x40041160,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB11CR,          0x40041168,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB11DL,          0x40041170,__READ_WRITE );
__IO_REG32(    CAN1MB11DH,          0x40041178,__READ_WRITE );
__IO_REG32_BIT(CAN1MB12ID,          0x40041180,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB12CR,          0x40041188,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB12DL,          0x40041190,__READ_WRITE );
__IO_REG32(    CAN1MB12DH,          0x40041198,__READ_WRITE );
__IO_REG32_BIT(CAN1MB13ID,          0x400411A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB13CR,          0x400411A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB13DL,          0x400411B0,__READ_WRITE );
__IO_REG32(    CAN1MB13DH,          0x400411B8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB14ID,          0x400411C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB14CR,          0x400411C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB14DL,          0x400411D0,__READ_WRITE );
__IO_REG32(    CAN1MB14DH,          0x400411D8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB15ID,          0x400411E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB15CR,          0x400411E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB15DL,          0x400411F0,__READ_WRITE );
__IO_REG32(    CAN1MB15DH,          0x400411F8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB16ID,          0x40041200,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB16CR,          0x40041208,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB16DL,          0x40041210,__READ_WRITE );
__IO_REG32(    CAN1MB16DH,          0x40041218,__READ_WRITE );
__IO_REG32_BIT(CAN1MB17ID,          0x40041220,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB17CR,          0x40041228,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB17DL,          0x40041230,__READ_WRITE );
__IO_REG32(    CAN1MB17DH,          0x40041238,__READ_WRITE );
__IO_REG32_BIT(CAN1MB18ID,          0x40041240,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB18CR,          0x40041248,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB18DL,          0x40041250,__READ_WRITE );
__IO_REG32(    CAN1MB18DH,          0x40041258,__READ_WRITE );
__IO_REG32_BIT(CAN1MB19ID,          0x40041260,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB19CR,          0x40041268,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB19DL,          0x40041270,__READ_WRITE );
__IO_REG32(    CAN1MB19DH,          0x40041278,__READ_WRITE );
__IO_REG32_BIT(CAN1MB20ID,          0x40041280,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB20CR,          0x40041288,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB20DL,          0x40041290,__READ_WRITE );
__IO_REG32(    CAN1MB20DH,          0x40041298,__READ_WRITE );
__IO_REG32_BIT(CAN1MB21ID,          0x400412A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB21CR,          0x400412A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB21DL,          0x400412B0,__READ_WRITE );
__IO_REG32(    CAN1MB21DH,          0x400412B8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB22ID,          0x400412C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB22CR,          0x400412C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB22DL,          0x400412D0,__READ_WRITE );
__IO_REG32(    CAN1MB22DH,          0x400412D8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB23ID,          0x400412E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB23CR,          0x400412E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB23DL,          0x400412F0,__READ_WRITE );
__IO_REG32(    CAN1MB23DH,          0x400412F8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB24ID,          0x40041300,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB24CR,          0x40041308,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB24DL,          0x40041310,__READ_WRITE );
__IO_REG32(    CAN1MB24DH,          0x40041318,__READ_WRITE );
__IO_REG32_BIT(CAN1MB25ID,          0x40041320,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB25CR,          0x40041328,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB25DL,          0x40041330,__READ_WRITE );
__IO_REG32(    CAN1MB25DH,          0x40041338,__READ_WRITE );
__IO_REG32_BIT(CAN1MB26ID,          0x40041340,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB26CR,          0x40041348,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB26DL,          0x40041350,__READ_WRITE );
__IO_REG32(    CAN1MB26DH,          0x40041358,__READ_WRITE );
__IO_REG32_BIT(CAN1MB27ID,          0x40041360,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB27CR,          0x40041368,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB27DL,          0x40041370,__READ_WRITE );
__IO_REG32(    CAN1MB27DH,          0x40041378,__READ_WRITE );
__IO_REG32_BIT(CAN1MB28ID,          0x40041380,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB28CR,          0x40041388,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB28DL,          0x40041390,__READ_WRITE );
__IO_REG32(    CAN1MB28DH,          0x40041398,__READ_WRITE );
__IO_REG32_BIT(CAN1MB29ID,          0x400413A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB29CR,          0x400413A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB29DL,          0x400413B0,__READ_WRITE );
__IO_REG32(    CAN1MB29DH,          0x400413B8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB30ID,          0x400413C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB30CR,          0x400413C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB30DL,          0x400413D0,__READ_WRITE );
__IO_REG32(    CAN1MB30DH,          0x400413D8,__READ_WRITE );
__IO_REG32_BIT(CAN1MB31ID,          0x400413E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN1MB31CR,          0x400413E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN1MB31DL,          0x400413F0,__READ_WRITE );
__IO_REG32(    CAN1MB31DH,          0x400413F8,__READ_WRITE );
__IO_REG32_BIT(CAN1MC,              0x40041400,__READ_WRITE ,__canmc_bits);
__IO_REG32_BIT(CAN1MD,              0x40041408,__READ_WRITE ,__canmd_bits);
__IO_REG32_BIT(CAN1TRS,             0x40041410,__READ_WRITE ,__cantrs_bits);
__IO_REG32_BIT(CAN1TRR,             0x40041418,__READ_WRITE ,__cantrr_bits);
__IO_REG32_BIT(CAN1TA,              0x40041420,__READ_WRITE ,__canta_bits);
__IO_REG32_BIT(CAN1AA,              0x40041428,__READ_WRITE ,__canaa_bits);
__IO_REG32_BIT(CAN1RMP,             0x40041430,__READ_WRITE ,__canrmp_bits);
__IO_REG32_BIT(CAN1RML,             0x40041438,__READ_WRITE ,__canrml_bits);
__IO_REG32_BIT(CAN1MCR,             0x40041450,__READ_WRITE ,__canmcr_bits);
__IO_REG32_BIT(CAN1GSR,             0x40041458,__READ       ,__cangsr_bits);
__IO_REG32_BIT(CAN1BCR1,            0x40041460,__READ_WRITE ,__canbcr1_bits);
__IO_REG32_BIT(CAN1BCR2,            0x40041468,__READ_WRITE ,__canbcr2_bits);
__IO_REG32_BIT(CAN1GIF,             0x40041470,__READ_WRITE ,__cangif_bits);
__IO_REG32_BIT(CAN1GIM,             0x40041478,__READ_WRITE ,__cangim_bits);
__IO_REG32_BIT(CAN1MBTIF,           0x40041480,__READ_WRITE ,__canmbtif_bits);
__IO_REG32_BIT(CAN1MBRIF,           0x40041488,__READ_WRITE ,__canmbrif_bits);
__IO_REG32_BIT(CAN1MBIM,            0x40041490,__READ_WRITE ,__canmbim_bits);
__IO_REG32_BIT(CAN1CDR,             0x40041498,__READ_WRITE ,__cancdr_bits);
__IO_REG32_BIT(CAN1RFP,             0x400414A0,__READ_WRITE ,__canrfp_bits);
__IO_REG32_BIT(CAN1CEC,             0x400414A8,__READ       ,__cancec_bits);
__IO_REG32_BIT(CAN1TSP,             0x400414B0,__READ_WRITE ,__cantsp_bits);
__IO_REG32_BIT(CAN1TSC,             0x400414B8,__READ_WRITE ,__cantsc_bits);
__IO_REG32_BIT(CAN1MB0AM,           0x400414C0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB1AM,           0x400414C8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB2AM,           0x400414D0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB3AM,           0x400414D8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB4AM,           0x400414E0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB5AM,           0x400414E8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB6AM,           0x400414F0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB7AM,           0x400414F8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB8AM,           0x40041500,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB9AM,           0x40041508,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB10AM,          0x40041510,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB11AM,          0x40041518,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB12AM,          0x40041520,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB13AM,          0x40041528,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB14AM,          0x40041530,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB15AM,          0x40041538,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB16AM,          0x40041540,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB17AM,          0x40041548,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB18AM,          0x40041550,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB19AM,          0x40041558,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB20AM,          0x40041560,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB21AM,          0x40041568,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB22AM,          0x40041570,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB23AM,          0x40041578,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB24AM,          0x40041580,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB25AM,          0x40041588,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB26AM,          0x40041590,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB27AM,          0x40041598,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB28AM,          0x400415A0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB29AM,          0x400415A8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB30AM,          0x400415B0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN1MB31AM,          0x400415B8,__READ_WRITE ,__canmbam_bits);
__IO_REG32(		 CAN1INTCRCLR,        0x40041800,__READ_WRITE );
__IO_REG32(		 CAN1INTCTCLR,        0x40041808,__READ_WRITE );
__IO_REG32(		 CAN1INTCGCLR,        0x40041810,__READ_WRITE );

/***************************************************************************
 **
 ** CAN2
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN2MB0ID,           0x40042000,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB0CR,           0x40042008,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB0DL,           0x40042010,__READ_WRITE );
__IO_REG32(    CAN2MB0DH,           0x40042018,__READ_WRITE );
__IO_REG32_BIT(CAN2MB1ID,           0x40042020,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB1CR,           0x40042028,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB1DL,           0x40042030,__READ_WRITE );
__IO_REG32(    CAN2MB1DH,           0x40042038,__READ_WRITE );
__IO_REG32_BIT(CAN2MB2ID,           0x40042040,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB2CR,           0x40042048,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB2DL,           0x40042050,__READ_WRITE );
__IO_REG32(    CAN2MB2DH,           0x40042058,__READ_WRITE );
__IO_REG32_BIT(CAN2MB3ID,           0x40042060,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB3CR,           0x40042068,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB3DL,           0x40042070,__READ_WRITE );
__IO_REG32(    CAN2MB3DH,           0x40042078,__READ_WRITE );
__IO_REG32_BIT(CAN2MB4ID,           0x40042080,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB4CR,           0x40042088,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB4DL,           0x40042090,__READ_WRITE );
__IO_REG32(    CAN2MB4DH,           0x40042098,__READ_WRITE );
__IO_REG32_BIT(CAN2MB5ID,           0x400420A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB5CR,           0x400420A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB5DL,           0x400420B0,__READ_WRITE );
__IO_REG32(    CAN2MB5DH,           0x400420B8,__READ_WRITE );
__IO_REG32_BIT(CAN2MB6ID,           0x400420C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB6CR,           0x400420C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB6DL,           0x400420D0,__READ_WRITE );
__IO_REG32(    CAN2MB6DH,           0x400420D8,__READ_WRITE );
__IO_REG32_BIT(CAN2MB7ID,           0x400420E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB7CR,           0x400420E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB7DL,           0x400420F0,__READ_WRITE );
__IO_REG32(    CAN2MB7DH,           0x400420F8,__READ_WRITE );
__IO_REG32_BIT(CAN2MB8ID,           0x40042100,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB8CR,           0x40042108,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB8DL,           0x40042110,__READ_WRITE );
__IO_REG32(    CAN2MB8DH,           0x40042118,__READ_WRITE );
__IO_REG32_BIT(CAN2MB9ID,           0x40042120,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB9CR,           0x40042128,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB9DL,           0x40042130,__READ_WRITE );
__IO_REG32(    CAN2MB9DH,           0x40042138,__READ_WRITE );
__IO_REG32_BIT(CAN2MB10ID,          0x40042140,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB10CR,          0x40042148,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB10DL,          0x40042150,__READ_WRITE );
__IO_REG32(    CAN2MB10DH,          0x40042158,__READ_WRITE );
__IO_REG32_BIT(CAN2MB11ID,          0x40042160,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB11CR,          0x40042168,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB11DL,          0x40042170,__READ_WRITE );
__IO_REG32(    CAN2MB11DH,          0x40042178,__READ_WRITE );
__IO_REG32_BIT(CAN2MB12ID,          0x40042180,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB12CR,          0x40042188,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB12DL,          0x40042190,__READ_WRITE );
__IO_REG32(    CAN2MB12DH,          0x40042198,__READ_WRITE );
__IO_REG32_BIT(CAN2MB13ID,          0x400421A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB13CR,          0x400421A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB13DL,          0x400421B0,__READ_WRITE );
__IO_REG32(    CAN2MB13DH,          0x400421B8,__READ_WRITE );
__IO_REG32_BIT(CAN2MB14ID,          0x400421C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB14CR,          0x400421C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB14DL,          0x400421D0,__READ_WRITE );
__IO_REG32(    CAN2MB14DH,          0x400421D8,__READ_WRITE );
__IO_REG32_BIT(CAN2MB15ID,          0x400421E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB15CR,          0x400421E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB15DL,          0x400421F0,__READ_WRITE );
__IO_REG32(    CAN2MB15DH,          0x400421F8,__READ_WRITE );
__IO_REG32_BIT(CAN2MB16ID,          0x40042200,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB16CR,          0x40042208,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB16DL,          0x40042210,__READ_WRITE );
__IO_REG32(    CAN2MB16DH,          0x40042218,__READ_WRITE );
__IO_REG32_BIT(CAN2MB17ID,          0x40042220,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB17CR,          0x40042228,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB17DL,          0x40042230,__READ_WRITE );
__IO_REG32(    CAN2MB17DH,          0x40042238,__READ_WRITE );
__IO_REG32_BIT(CAN2MB18ID,          0x40042240,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB18CR,          0x40042248,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB18DL,          0x40042250,__READ_WRITE );
__IO_REG32(    CAN2MB18DH,          0x40042258,__READ_WRITE );
__IO_REG32_BIT(CAN2MB19ID,          0x40042260,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB19CR,          0x40042268,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB19DL,          0x40042270,__READ_WRITE );
__IO_REG32(    CAN2MB19DH,          0x40042278,__READ_WRITE );
__IO_REG32_BIT(CAN2MB20ID,          0x40042280,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB20CR,          0x40042288,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB20DL,          0x40042290,__READ_WRITE );
__IO_REG32(    CAN2MB20DH,          0x40042298,__READ_WRITE );
__IO_REG32_BIT(CAN2MB21ID,          0x400422A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB21CR,          0x400422A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB21DL,          0x400422B0,__READ_WRITE );
__IO_REG32(    CAN2MB21DH,          0x400422B8,__READ_WRITE );
__IO_REG32_BIT(CAN2MB22ID,          0x400422C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB22CR,          0x400422C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB22DL,          0x400422D0,__READ_WRITE );
__IO_REG32(    CAN2MB22DH,          0x400422D8,__READ_WRITE );
__IO_REG32_BIT(CAN2MB23ID,          0x400422E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB23CR,          0x400422E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB23DL,          0x400422F0,__READ_WRITE );
__IO_REG32(    CAN2MB23DH,          0x400422F8,__READ_WRITE );
__IO_REG32_BIT(CAN2MB24ID,          0x40042300,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB24CR,          0x40042308,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB24DL,          0x40042310,__READ_WRITE );
__IO_REG32(    CAN2MB24DH,          0x40042318,__READ_WRITE );
__IO_REG32_BIT(CAN2MB25ID,          0x40042320,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB25CR,          0x40042328,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB25DL,          0x40042330,__READ_WRITE );
__IO_REG32(    CAN2MB25DH,          0x40042338,__READ_WRITE );
__IO_REG32_BIT(CAN2MB26ID,          0x40042340,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB26CR,          0x40042348,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB26DL,          0x40042350,__READ_WRITE );
__IO_REG32(    CAN2MB26DH,          0x40042358,__READ_WRITE );
__IO_REG32_BIT(CAN2MB27ID,          0x40042360,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB27CR,          0x40042368,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB27DL,          0x40042370,__READ_WRITE );
__IO_REG32(    CAN2MB27DH,          0x40042378,__READ_WRITE );
__IO_REG32_BIT(CAN2MB28ID,          0x40042380,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB28CR,          0x40042388,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB28DL,          0x40042390,__READ_WRITE );
__IO_REG32(    CAN2MB28DH,          0x40042398,__READ_WRITE );
__IO_REG32_BIT(CAN2MB29ID,          0x400423A0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB29CR,          0x400423A8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB29DL,          0x400423B0,__READ_WRITE );
__IO_REG32(    CAN2MB29DH,          0x400423B8,__READ_WRITE );
__IO_REG32_BIT(CAN2MB30ID,          0x400423C0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB30CR,          0x400423C8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB30DL,          0x400423D0,__READ_WRITE );
__IO_REG32(    CAN2MB30DH,          0x400423D8,__READ_WRITE );
__IO_REG32_BIT(CAN2MB31ID,          0x400423E0,__READ_WRITE ,__canmbid_bits);
__IO_REG32_BIT(CAN2MB31CR,          0x400423E8,__READ_WRITE ,__canmbtcr_bits);
__IO_REG32(    CAN2MB31DL,          0x400423F0,__READ_WRITE );
__IO_REG32(    CAN2MB31DH,          0x400423F8,__READ_WRITE );
__IO_REG32_BIT(CAN2MC,              0x40042400,__READ_WRITE ,__canmc_bits);
__IO_REG32_BIT(CAN2MD,              0x40042408,__READ_WRITE ,__canmd_bits);
__IO_REG32_BIT(CAN2TRS,             0x40042410,__READ_WRITE ,__cantrs_bits);
__IO_REG32_BIT(CAN2TRR,             0x40042418,__READ_WRITE ,__cantrr_bits);
__IO_REG32_BIT(CAN2TA,              0x40042420,__READ_WRITE ,__canta_bits);
__IO_REG32_BIT(CAN2AA,              0x40042428,__READ_WRITE ,__canaa_bits);
__IO_REG32_BIT(CAN2RMP,             0x40042430,__READ_WRITE ,__canrmp_bits);
__IO_REG32_BIT(CAN2RML,             0x40042438,__READ_WRITE ,__canrml_bits);
__IO_REG32_BIT(CAN2MCR,             0x40042450,__READ_WRITE ,__canmcr_bits);
__IO_REG32_BIT(CAN2GSR,             0x40042458,__READ       ,__cangsr_bits);
__IO_REG32_BIT(CAN2BCR1,            0x40042460,__READ_WRITE ,__canbcr1_bits);
__IO_REG32_BIT(CAN2BCR2,            0x40042468,__READ_WRITE ,__canbcr2_bits);
__IO_REG32_BIT(CAN2GIF,             0x40042470,__READ_WRITE ,__cangif_bits);
__IO_REG32_BIT(CAN2GIM,             0x40042478,__READ_WRITE ,__cangim_bits);
__IO_REG32_BIT(CAN2MBTIF,           0x40042480,__READ_WRITE ,__canmbtif_bits);
__IO_REG32_BIT(CAN2MBRIF,           0x40042488,__READ_WRITE ,__canmbrif_bits);
__IO_REG32_BIT(CAN2MBIM,            0x40042490,__READ_WRITE ,__canmbim_bits);
__IO_REG32_BIT(CAN2CDR,             0x40042498,__READ_WRITE ,__cancdr_bits);
__IO_REG32_BIT(CAN2RFP,             0x400424A0,__READ_WRITE ,__canrfp_bits);
__IO_REG32_BIT(CAN2CEC,             0x400424A8,__READ       ,__cancec_bits);
__IO_REG32_BIT(CAN2TSP,             0x400424B0,__READ_WRITE ,__cantsp_bits);
__IO_REG32_BIT(CAN2TSC,             0x400424B8,__READ_WRITE ,__cantsc_bits);
__IO_REG32_BIT(CAN2MB0AM,           0x400424C0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB1AM,           0x400424C8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB2AM,           0x400424D0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB3AM,           0x400424D8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB4AM,           0x400424E0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB5AM,           0x400424E8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB6AM,           0x400424F0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB7AM,           0x400424F8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB8AM,           0x40042500,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB9AM,           0x40042508,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB10AM,          0x40042510,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB11AM,          0x40042518,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB12AM,          0x40042520,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB13AM,          0x40042528,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB14AM,          0x40042530,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB15AM,          0x40042538,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB16AM,          0x40042540,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB17AM,          0x40042548,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB18AM,          0x40042550,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB19AM,          0x40042558,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB20AM,          0x40042560,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB21AM,          0x40042568,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB22AM,          0x40042570,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB23AM,          0x40042578,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB24AM,          0x40042580,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB25AM,          0x40042588,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB26AM,          0x40042590,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB27AM,          0x40042598,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB28AM,          0x400425A0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB29AM,          0x400425A8,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB30AM,          0x400425B0,__READ_WRITE ,__canmbam_bits);
__IO_REG32_BIT(CAN2MB31AM,          0x400425B8,__READ_WRITE ,__canmbam_bits);
__IO_REG32(		 CAN2INTCRCLR,        0x40042800,__READ_WRITE );
__IO_REG32(		 CAN2INTCTCLR,        0x40042808,__READ_WRITE );
__IO_REG32(		 CAN2INTCGCLR,        0x40042810,__READ_WRITE );

/***************************************************************************
 **
 ** ESEI0
 **
 ***************************************************************************/
__IO_REG32_BIT(ESEI0_MCR,        		0x40043000,__READ_WRITE ,__esei_mcr_bits);
__IO_REG32_BIT(ESEI0_CR0,        		0x40043004,__READ_WRITE ,__esei_cr0_bits);
__IO_REG32_BIT(ESEI0_CR1,        		0x40043008,__READ_WRITE ,__esei_cr1_bits);
__IO_REG32_BIT(ESEI0_FSR,        		0x4004300C,__READ_WRITE ,__esei_fsr_bits);
__IO_REG32_BIT(ESEI0_SSR,        		0x40043010,__READ_WRITE ,__esei_ssr_bits);
__IO_REG32_BIT(ESEI0_SR,         		0x40043014,__READ       ,__esei_sr_bits);
__IO_REG32_BIT(ESEI0_DR,         		0x40043018,__READ_WRITE ,__esei_dr_bits);
__IO_REG32_BIT(ESEI0_RSR,        		0x4004301C,__READ				,__esei_rsr_bits);
__IO_REG32_BIT(ESEI0_FLR,        		0x40043020,__READ				,__esei_flr_bits);
__IO_REG32_BIT(ESEI0_ILR,        		0x40043024,__READ_WRITE ,__esei_ilr_bits);
__IO_REG32_BIT(ESEI0_PR,         		0x40043028,__READ_WRITE ,__esei_pr_bits);
__IO_REG32_BIT(ESEI0_LCR,        		0x4004302C,__READ_WRITE ,__esei_lcr_bits);
__IO_REG32_BIT(ESEI0_DER,        		0x40043030,__READ_WRITE ,__esei_der_bits);
#define ESEI0_DERW        ESEI0_DER
#define ESEI0_DERW_bit    ESEI0_DER_bit.__write
__IO_REG32_BIT(ESEI0_EICR,       		0x40043800,__WRITE 			,__esei_eicr_bits);
__IO_REG32_BIT(ESEI0_RICR,       		0x40043804,__WRITE 			,__esei_ricr_bits);
__IO_REG32_BIT(ESEI0_TICR,       		0x40043808,__WRITE 			,__esei_ticr_bits);

/***************************************************************************
 **
 ** ESEI1
 **
 ***************************************************************************/
__IO_REG32_BIT(ESEI1_MCR,        		0x40044000,__READ_WRITE ,__esei_mcr_bits);
__IO_REG32_BIT(ESEI1_CR0,        		0x40044004,__READ_WRITE ,__esei_cr0_bits);
__IO_REG32_BIT(ESEI1_CR1,        		0x40044008,__READ_WRITE ,__esei_cr1_bits);
__IO_REG32_BIT(ESEI1_FSR,        		0x4004400C,__READ_WRITE ,__esei_fsr_bits);
__IO_REG32_BIT(ESEI1_SSR,        		0x40044010,__READ_WRITE ,__esei_ssr_bits);
__IO_REG32_BIT(ESEI1_SR,         		0x40044014,__READ       ,__esei_sr_bits);
__IO_REG32_BIT(ESEI1_DR,         		0x40044018,__READ_WRITE ,__esei_dr_bits);
__IO_REG32_BIT(ESEI1_RSR,        		0x4004401C,__READ				,__esei_rsr_bits);
__IO_REG32_BIT(ESEI1_FLR,        		0x40044020,__READ				,__esei_flr_bits);
__IO_REG32_BIT(ESEI1_ILR,        		0x40044024,__READ_WRITE ,__esei_ilr_bits);
__IO_REG32_BIT(ESEI1_PR,         		0x40044028,__READ_WRITE ,__esei_pr_bits);
__IO_REG32_BIT(ESEI1_LCR,        		0x4004402C,__READ_WRITE ,__esei_lcr_bits);
__IO_REG32_BIT(ESEI1_DER,        		0x40044030,__READ_WRITE ,__esei_der_bits);
#define ESEI1_DERW        ESEI1_DER
#define ESEI1_DERW_bit    ESEI1_DER_bit.__write
__IO_REG32_BIT(ESEI1_EICR,       		0x40044800,__WRITE 			,__esei_eicr_bits);
__IO_REG32_BIT(ESEI1_RICR,       		0x40044804,__WRITE 			,__esei_ricr_bits);
__IO_REG32_BIT(ESEI1_TICR,       		0x40044808,__WRITE 			,__esei_ticr_bits);

/***************************************************************************
 **
 ** CRC
 **
 ***************************************************************************/
__IO_REG32(		 CRCDIN,             	0x4001C000,__READ_WRITE );
__IO_REG32_BIT(CRCTYP,          	  0x4001C014,__READ_WRITE ,__crctyp_bits);
__IO_REG32_BIT(CRCRST,             	0x4001C028,__READ_WRITE ,__crcrst_bits);

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
__IO_REG32_BIT(CARSRC,            	0x40022020,__READ       ,__rate_bits);
__IO_REG32_BIT(CARREF,            	0x40022024,__READ				,__rate_bits);
__IO_REG32_BIT(INSDAT0,            	0x40022028,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT1,            	0x4002202C,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT2,            	0x40022030,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT3,            	0x40022034,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT4,            	0x40022038,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT5,            	0x4002203C,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT6,            	0x40022040,__READ_WRITE ,__insdat_bits);
__IO_REG32_BIT(INSDAT7,            	0x40022044,__READ_WRITE ,__insdat_bits);

/***************************************************************************
 **
 ** VE
 **
 ***************************************************************************/
__IO_REG32_BIT(VEEN,                0x40050000, __READ_WRITE , __veen_bits);
__IO_REG32_BIT(VECPURUNTRG,         0x40050004, __WRITE      , __cpuruntrg_bits);
__IO_REG32_BIT(VETASKAPP,           0x40050008, __READ_WRITE , __taskapp_bits);
__IO_REG32_BIT(VEACTSCH,            0x4005000C, __READ_WRITE , __actsch_bits);
__IO_REG32_BIT(VEREPTIME,           0x40050010, __READ_WRITE , __reptime_bits);
__IO_REG32_BIT(VETRGMODE,           0x40050014, __READ_WRITE , __trgmode_bits);
__IO_REG32_BIT(VEERRINTEN,          0x40050018, __READ_WRITE , __errinten_bits);
__IO_REG32_BIT(VECOMPEND,           0x4005001C, __WRITE      , __compend_bits);
__IO_REG32_BIT(VEERRDET,            0x40050020, __READ       , __errdet_bits);
__IO_REG32_BIT(VESCHTASKRUN,        0x40050024, __READ       , __schtaskrun_bits);
__IO_REG32(    VETMPREG0,           0x4005002C, __READ_WRITE );
__IO_REG32(    VETMPREG1,           0x40050030, __READ_WRITE );
__IO_REG32(    VETMPREG2,           0x40050034, __READ_WRITE );
__IO_REG32(    VETMPREG3,           0x40050038, __READ_WRITE );
__IO_REG32(    VETMPREG4,           0x4005003C, __READ_WRITE );
__IO_REG32(    VETMPREG5,           0x40050040, __READ_WRITE );
__IO_REG32_BIT(VEMCTLF,             0x40050044, __READ_WRITE , __mctlfx_bits);
__IO_REG32_BIT(VEMODE,              0x40050048, __READ_WRITE , __modex_bits);
__IO_REG32_BIT(VEFMODE,             0x4005004C, __READ_WRITE , __fmodex_bits);
__IO_REG32_BIT(VETPWM,              0x40050050, __READ_WRITE , __tpwmx_bits);
__IO_REG32_BIT(VEOMEGA,             0x40050054, __READ_WRITE , __omegax_bits);
__IO_REG32_BIT(VETHETA,             0x40050058, __READ_WRITE , __thetax_bits);
__IO_REG32_BIT(VEIDREF,             0x4005005C, __READ_WRITE , __idrefx_bits);
__IO_REG32_BIT(VEIQREF,             0x40050060, __READ_WRITE , __iqrefx_bits);
__IO_REG32_BIT(VEVD,                0x40050064, __READ_WRITE , __vdx_bits);
__IO_REG32_BIT(VEVQ,                0x40050068, __READ_WRITE , __vqx_bits);
__IO_REG32_BIT(VECIDKI,             0x4005006C, __READ_WRITE , __cidkix_bits);
__IO_REG32_BIT(VECIDKP,             0x40050070, __READ_WRITE , __cidkpx_bits);
__IO_REG32_BIT(VECIQKI,             0x40050074, __READ_WRITE , __ciqkix_bits);
__IO_REG32_BIT(VECIQKP,             0x40050078, __READ_WRITE , __ciqkpx_bits);
__IO_REG32_BIT(VEVDIH,              0x4005007C, __READ_WRITE , __vdihx_bits);
__IO_REG32_BIT(VEVDILH,             0x40050080, __READ_WRITE , __vdilhx_bits);
__IO_REG32_BIT(VEVQIH,              0x40050084, __READ_WRITE , __vqihx_bits);
__IO_REG32_BIT(VEVQILH,             0x40050088, __READ_WRITE , __vqilhx_bits);
__IO_REG32_BIT(VEMDPRD,             0x40050090, __READ_WRITE , __vmdprdx_bits);
__IO_REG32_BIT(VEMINPLS,            0x40050094, __READ_WRITE , __minplsx_bits);
__IO_REG32_BIT(VETRGCRC,            0x40050098, __READ_WRITE , __trgcrcx_bits);
__IO_REG32_BIT(VECOS,               0x400500A0, __READ_WRITE , __cosx_bits);
__IO_REG32_BIT(VESIN,               0x400500A4, __READ_WRITE , __sinx_bits);
__IO_REG32_BIT(VECOSM,              0x400500A8, __READ_WRITE , __cosmx_bits);
__IO_REG32_BIT(VESINM,              0x400500AC, __READ_WRITE , __sinmx_bits);
__IO_REG32_BIT(VESECTOR,            0x400500B0, __READ_WRITE , __sectorx_bits);
__IO_REG32_BIT(VESECTORM,           0x400500B4, __READ_WRITE , __sectormx_bits);
__IO_REG32_BIT(VEIAO,               0x400500B8, __READ_WRITE , __iaox_bits);
__IO_REG32_BIT(VEIBO,               0x400500BC, __READ_WRITE , __ibox_bits);
__IO_REG32_BIT(VEICO,               0x400500C0, __READ_WRITE , __icox_bits);
__IO_REG32_BIT(VEIAADC,             0x400500C4, __READ_WRITE , __iaadcx_bits);
__IO_REG32_BIT(VEIBADC,             0x400500C8, __READ_WRITE , __ibadcx_bits);
__IO_REG32_BIT(VEICADC,             0x400500CC, __READ_WRITE , __icadcx_bits);
__IO_REG32_BIT(VEVDC,               0x400500D0, __READ_WRITE , __vdcx_bits);
__IO_REG32_BIT(VEID,                0x400500D4, __READ_WRITE , __idx_bits);
__IO_REG32_BIT(VEIQ,                0x400500D8, __READ_WRITE , __iqx_bits);     
__IO_REG32_BIT(VECMPU0,             0x4005017C, __READ_WRITE , __vcmpux_bits);
__IO_REG32_BIT(VECMPV0,             0x40050180, __READ_WRITE , __vcmpvx_bits);
__IO_REG32_BIT(VECMPW0,             0x40050184, __READ_WRITE , __vcmpwx_bits);
__IO_REG32_BIT(VEOUTCR,             0x40050188, __READ_WRITE , __outcrx_bits);
__IO_REG32_BIT(VETRGCMPA,           0x4005018C, __READ_WRITE , __vtrgcmpax_bits);
__IO_REG32_BIT(VETRGCMPB,           0x40050190, __READ_WRITE , __vtrgcmpbx_bits);
__IO_REG32_BIT(VEEMGRS,             0x40050198, __WRITE      , __emgrsx_bits);   
__IO_REG32_BIT(VETSKPGM0,           0x400501C0, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGM1,           0x400501C4, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGM2,           0x400501C8, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGM3,           0x400501CC, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGM4,           0x400501D0, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGM5,           0x400501D4, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGM6,           0x400501D8, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGM7,           0x400501DC, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGM8,           0x400501E0, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGM9,           0x400501E4, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGMA,           0x400501E8, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGMB,           0x400501EC, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGMC,           0x400501F0, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGMD,           0x400501F4, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGME,           0x400501F8, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKPGMF,           0x400501FC, __READ_WRITE , __tskpgmx_bits);
__IO_REG32_BIT(VETSKINITP,          0x40050240, __READ_WRITE , __tskinitp_bits);

/***************************************************************************
 **
 ** PMD
 **
 ***************************************************************************/
__IO_REG32_BIT(PMD_RATECR0,         0x40010000,__READ_WRITE ,__pmd_ratecr0_bits);
__IO_REG32_BIT(PMD_RCCCR0,          0x40010004,__READ_WRITE ,__pmd_rcccr0_bits);
__IO_REG32_BIT(PMD_RATETSEL0,       0x40010008,__READ_WRITE ,__pmd_ratetsel0_bits);
__IO_REG32_BIT(PMD_RATE0,           0x4001000C,__READ_WRITE ,__pmd_rate0_bits);
__IO_REG32(    PMD_RCOUNT0,         0x40010010,__READ_WRITE );
__IO_REG32_BIT(PMD_CARSET0,         0x40010020,__READ_WRITE ,__pmd_carset0_bits);
__IO_REG32_BIT(PMD_CARSIF0,         0x40010024,__READ_WRITE ,__pmd_carsif0_bits);
__IO_REG32_BIT(PMD_CMPU0,           0x40010028,__READ_WRITE ,__pmd_cmpu0_bits);
__IO_REG32_BIT(PMD_CMPV0,           0x4001002C,__READ_WRITE ,__pmd_cmpv0_bits);
__IO_REG32_BIT(PMD_CMPW0,           0x40010030,__READ_WRITE ,__pmd_cmpw0_bits);
__IO_REG32_BIT(PMD_NOKORD0,         0x40010034,__READ       ,__pmd_nokord0_bits);
__IO_REG32_BIT(PMD_CARH0,           0x40010038,__READ       ,__pmd_carh0_bits);
__IO_REG32_BIT(PMD_CARL0,           0x4001003C,__READ       ,__pmd_carl0_bits);
__IO_REG32_BIT(PMD_CARSET1,         0x40010040,__READ_WRITE ,__pmd_carset1_bits);
__IO_REG32_BIT(PMD_CARSIF1,         0x40010044,__READ_WRITE ,__pmd_carsif1_bits);
__IO_REG32_BIT(PMD_TRGCMPA1,        0x40010048,__READ_WRITE ,__pmd_trgcmpa1_bits);
__IO_REG32_BIT(PMD_TRGCMPB1,        0x4001004C,__READ_WRITE ,__pmd_trgcmpb1_bits);
__IO_REG32_BIT(PMD_NOKORD1,         0x40010050,__READ       ,__pmd_nokord1_bits);
__IO_REG32_BIT(PMD_CARH1,           0x40010054,__READ       ,__pmd_carh1_bits);
__IO_REG32_BIT(PMD_RATECR1,         0x40010060,__READ_WRITE ,__pmd_ratecr1_bits);
__IO_REG32_BIT(PMD_RCCCR1,          0x40010064,__READ_WRITE ,__pmd_rcccr1_bits);
__IO_REG32_BIT(PMD_RATETSEL1,       0x40010068,__READ_WRITE ,__pmd_ratetsel1_bits);
__IO_REG32_BIT(PMD_RATE1,           0x4001006C,__READ_WRITE ,__pmd_rate1_bits);
__IO_REG32(    PMD_RCOUNT1,         0x40010070,__READ_WRITE );
__IO_REG32_BIT(PMD_CARSET2,         0x40010080,__READ_WRITE ,__pmd_carset2_bits);
__IO_REG32_BIT(PMD_CARSIF2,         0x40010084,__READ_WRITE ,__pmd_carsif2_bits);
__IO_REG32_BIT(PMD_CPWMA2,          0x40010088,__READ_WRITE ,__pmd_cpwma2_bits);
__IO_REG32_BIT(PMD_CPWMB2,          0x4001008C,__READ_WRITE ,__pmd_cpwmb2_bits);
__IO_REG32_BIT(PMD_NOKORD2,         0x40010090,__READ       ,__pmd_nokord2_bits);
__IO_REG32_BIT(PMD_CARH2,           0x40010094,__READ       ,__pmd_carh2_bits);
__IO_REG32_BIT(PMD_CARL2,           0x40010098,__READ       ,__pmd_carl2_bits);
__IO_REG32_BIT(PMD_CARCNT,          0x400100A0,__READ_WRITE ,__pmd_carcnt_bits);
__IO_REG32_BIT(PMD_SIFCNT,          0x400100A4,__READ_WRITE ,__pmd_sifcnt_bits);
__IO_REG32_BIT(PMD_CMPCNT,          0x400100A8,__READ_WRITE ,__pmd_cmpcnt_bits);
__IO_REG32_BIT(PMD_PO_DTR,          0x40010100,__READ_WRITE ,__pmd_po_dtr_bits);
__IO_REG32_BIT(PMD_PO_MPR,          0x40010108,__READ_WRITE ,__pmd_po_mpr_bits);
__IO_REG32_BIT(PMD_PO_MDEN,         0x40010110,__READ_WRITE ,__pmd_po_mden_bits);
__IO_REG32_BIT(PMD_PO_PORTMD,       0x40010114,__READ_WRITE ,__pmd_po_portmd_bits);
__IO_REG32_BIT(PMD_PO_MDCR,         0x40010118,__READ_WRITE ,__pmd_po_mdcr_bits);
__IO_REG32_BIT(PMD_PO_MDOUT,        0x4001011C,__READ_WRITE ,__pmd_po_mdout_bits);
__IO_REG32_BIT(PMD_PO_MDPOT,        0x40010120,__READ_WRITE ,__pmd_po_mdpot_bits);
__IO_REG32_BIT(PMD_PO_EMGREL,       0x40010124,__WRITE      ,__pmd_po_emgrel_bits);
__IO_REG32_BIT(PMD_PO_EMGCR,        0x40010128,__READ_WRITE ,__pmd_po_emgcr_bits);
__IO_REG32_BIT(PMD_PO_EMGSTA,       0x4001012C,__READ       ,__pmd_po_emgsta_bits);
__IO_REG32_BIT(PMD_MSET,            0x40010200,__READ_WRITE ,__pmd_mset_bits);
__IO_REG32_BIT(PMD_WLOAD,           0x40010204,__READ_WRITE ,__pmd_wload_bits);
__IO_REG32(    PMD_WS,              0x40010208,__READ_WRITE );
__IO_REG32_BIT(PMD_QSH,             0x4001020C,__READ       ,__pmd_qsh_bits);
__IO_REG32(    PMD_QSML,            0x40010210,__READ_WRITE );
__IO_REG32_BIT(PMD_QT,              0x40010214,__READ_WRITE ,__pmd_qt_bits);
__IO_REG32_BIT(PMD_SPWMQTU,         0x40010218,__READ_WRITE ,__pmd_spwmqtu_bits);
__IO_REG32_BIT(PMD_SPWMQTV,         0x4001021C,__READ_WRITE ,__pmd_spwmqtv_bits);
__IO_REG32_BIT(PMD_SPWMQTW,         0x40010220,__READ_WRITE ,__pmd_spwmqtw_bits);
__IO_REG32_BIT(PMD_Q1A,             0x40010224,__READ       ,__pmd_q1a_bits);
__IO_REG32_BIT(PMD_Q1B,             0x40010228,__READ       ,__pmd_q1b_bits);
__IO_REG32_BIT(PMD_Q1C,             0x4001022C,__READ       ,__pmd_q1c_bits);
__IO_REG32_BIT(PMD_QRWDT,           0x40010300,__READ_WRITE ,__pmd_qrwdt_bits);
__IO_REG32_BIT(PMD_QRCR,            0x40010304,__READ_WRITE ,__pmd_qrcr_bits);
__IO_REG32_BIT(PMD_QRSW,            0x40010308,__READ_WRITE ,__pmd_qrsw_bits);
__IO_REG32_BIT(PMD_QROFFSET,        0x4001030C,__READ_WRITE ,__pmd_qroffset_bits);
__IO_REG32_BIT(PMD_QRCMP,           0x40010310,__READ_WRITE ,__pmd_qrcmp_bits);
__IO_REG32_BIT(PMD_QRADD,           0x40010314,__READ_WRITE ,__pmd_qradd_bits);
__IO_REG32_BIT(PMD_QRCOUNT,         0x40010318,__READ_WRITE ,__pmd_qrcount_bits);
__IO_REG32_BIT(PMD_QROUT,           0x4001031C,__READ_WRITE ,__pmd_qrout_bits);
__IO_REG32_BIT(PMD_THETAAD,         0x40010320,__READ_WRITE ,__pmd_thetaad_bits);
__IO_REG32_BIT(PMD_THETAOUT,        0x40010324,__READ_WRITE ,__pmd_thetaout_bits);
__IO_REG32_BIT(PMD_QRWCR,           0x40010328,__WRITE      ,__pmd_qrwcr_bits);
__IO_REG32_BIT(PMD_DPWMTSEL,        0x40010400,__READ_WRITE ,__pmd_dpwmtsel_bits);
__IO_REG32_BIT(PMD_DIRPWM,          0x40010404,__READ_WRITE ,__pmd_dirpwm_bits);
__IO_REG32_BIT(PMD_VESW,            0x40010420,__READ_WRITE ,__pmd_vesw_bits);
__IO_REG32_BIT(PMD_PO_DSW,          0x40010424,__READ_WRITE ,__pmd_po_dsw_bits);
__IO_REG32_BIT(PMD_CO_DSEL0,        0x40010428,__READ_WRITE ,__pmd_co_dsel0_bits);
__IO_REG32_BIT(PMD_CO_DSEL1,        0x4001042C,__READ_WRITE ,__pmd_co_dsel1_bits);
__IO_REG32_BIT(PMD_ADTRGSEL,        0x40010430,__READ_WRITE ,__pmd_adtrgsel_bits);
__IO_REG32_BIT(PMD_RATESW,          0x40010434,__READ_WRITE ,__pmd_ratesw_bits);
__IO_REG32_BIT(PMD_PCSR0,           0x40010500,__READ_WRITE ,__pmd_pcsr0_bits);
__IO_REG32_BIT(PMD_PDOUT0,          0x40010504,__READ_WRITE ,__pmd_pdout0_bits);
__IO_REG32_BIT(PMD_PWMC0,           0x40010508,__READ_WRITE ,__pmd_pwmc0_bits);
__IO_REG32_BIT(PMD_CNT0,            0x4001050C,__WRITE      ,__pmd_cnt0_bits);
__IO_REG32_BIT(PMD_PCSR1,           0x40010510,__READ_WRITE ,__pmd_pcsr1_bits);
__IO_REG32_BIT(PMD_PDOUT1,          0x40010514,__READ_WRITE ,__pmd_pdout1_bits);
__IO_REG32_BIT(PMD_PWMC1,           0x40010518,__READ_WRITE ,__pmd_pwmc1_bits);
__IO_REG32_BIT(PMD_CNT1,            0x4001051C,__WRITE      ,__pmd_cnt1_bits);
__IO_REG32_BIT(PMD_SPWMQA,          0x40010600,__READ_WRITE ,__pmd_spwmqa_bits);
__IO_REG32_BIT(PMD_SPWMQB,          0x40010604,__READ_WRITE ,__pmd_spwmqb_bits);
__IO_REG32_BIT(PMD_SPWMQC,          0x40010608,__READ_WRITE ,__pmd_spwmqc_bits);
__IO_REG32_BIT(PMD_SPWMQD,          0x4001060C,__READ_WRITE ,__pmd_spwmqd_bits);
__IO_REG32_BIT(PMD_SPWMQE,          0x40010610,__READ_WRITE ,__pmd_spwmqe_bits);
__IO_REG32_BIT(PMD_SPWMQF,          0x40010614,__READ_WRITE ,__pmd_spwmqf_bits);
__IO_REG32_BIT(PMD_SPWMQG,          0x40010618,__READ_WRITE ,__pmd_spwmqg_bits);
__IO_REG32_BIT(PMD_SPWMQH,          0x4001061C,__READ_WRITE ,__pmd_spwmqh_bits);
__IO_REG32_BIT(PMD_SPWMQI,          0x40010620,__READ_WRITE ,__pmd_spwmqi_bits);
__IO_REG32_BIT(PMD_SPWMQJ,          0x40010624,__READ_WRITE ,__pmd_spwmqj_bits);
__IO_REG32_BIT(PMD_SPWMQK,          0x40010628,__READ_WRITE ,__pmd_spwmqk_bits);
__IO_REG32_BIT(PMD_SPWMQL,          0x4001062C,__READ_WRITE ,__pmd_spwmql_bits);
__IO_REG32_BIT(PMD_SPWMQM,          0x40010630,__READ_WRITE ,__pmd_spwmqm_bits);
__IO_REG32_BIT(PMD_CO_DTR0,         0x40010700,__READ_WRITE ,__pmd_co_dtr0_bits);
__IO_REG32_BIT(PMD_CO_MPR0,         0x40010708,__READ_WRITE ,__pmd_co_mpr0_bits);
__IO_REG32_BIT(PMD_CO_MDEN0,        0x40010710,__READ_WRITE ,__pmd_co_mden0_bits);
__IO_REG32_BIT(PMD_CO_PORTMD0,      0x40010714,__READ_WRITE ,__pmd_co_portmd0_bits);
__IO_REG32_BIT(PMD_CO_MDCR0,        0x40010718,__READ_WRITE ,__pmd_co_mdcr0_bits);
__IO_REG32_BIT(PMD_CO_MDOUT0,       0x4001071C,__READ_WRITE ,__pmd_co_mdout0_bits);
__IO_REG32_BIT(PMD_CO_MDPOT0,       0x40010720,__READ_WRITE ,__pmd_co_mdpot0_bits);
__IO_REG32_BIT(PMD_CO_EMGREL0,      0x40010724,__WRITE      ,__pmd_co_emgrel0_bits);
__IO_REG32_BIT(PMD_CO_EMGCR0,       0x40010728,__READ_WRITE ,__pmd_co_emgcr0_bits);
__IO_REG32_BIT(PMD_CO_EMGSTA0,      0x4001072C,__READ       ,__pmd_co_emgsta0_bits);
__IO_REG32_BIT(PMD_CO_DTR1,         0x40010800,__READ_WRITE ,__pmd_co_dtr0_bits);
__IO_REG32_BIT(PMD_CO_MPR1,         0x40010808,__READ_WRITE ,__pmd_co_mpr0_bits);
__IO_REG32_BIT(PMD_CO_MDEN1,        0x40010810,__READ_WRITE ,__pmd_co_mden0_bits);
__IO_REG32_BIT(PMD_CO_PORTMD1,      0x40010814,__READ_WRITE ,__pmd_co_portmd0_bits);
__IO_REG32_BIT(PMD_CO_MDCR1,        0x40010818,__READ_WRITE ,__pmd_co_mdcr0_bits);
__IO_REG32_BIT(PMD_CO_MDOUT1,       0x4001081C,__READ_WRITE ,__pmd_co_mdout0_bits);
__IO_REG32_BIT(PMD_CO_MDPOT1,       0x40010820,__READ_WRITE ,__pmd_co_mdpot0_bits);
__IO_REG32_BIT(PMD_CO_EMGREL1,      0x40010824,__WRITE      ,__pmd_co_emgrel0_bits);
__IO_REG32_BIT(PMD_CO_EMGCR1,       0x40010828,__READ_WRITE ,__pmd_co_emgcr0_bits);
__IO_REG32_BIT(PMD_CO_EMGSTA1,      0x4001082C,__READ_WRITE ,__pmd_co_emgsta1_bits);
__IO_REG32_BIT(PMD_NCMP,            0x40010900,__READ_WRITE ,__pmd_ncmp_bits);
__IO_REG32_BIT(PMD_SCMP,            0x40010904,__READ_WRITE ,__pmd_scmp_bits);
__IO_REG32_BIT(PMD_P120CR,          0x40010908,__READ_WRITE ,__pmd_p120cr_bits);

/***************************************************************************
 **
 ** ADC0
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC0RSLT0,           0x4002D000,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT1,           0x4002D004,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT2,           0x4002D008,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT3,           0x4002D00C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT4,           0x4002D010,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT5,           0x4002D014,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT6,           0x4002D018,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT7,           0x4002D01C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT8,           0x4002D020,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT9,           0x4002D024,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT10,          0x4002D028,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT11,          0x4002D02C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT12,          0x4002D030,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0RSLT13,          0x4002D034,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC0SETI0,          	0x4002D038,__READ_WRITE ,__adcseti0_bits);
__IO_REG32_BIT(ADC0SETI1,         	0x4002D03C,__READ_WRITE ,__adcseti1_bits);
__IO_REG32_BIT(ADC0SETT,         		0x4002D040,__READ_WRITE ,__adcsett_bits);
__IO_REG32_BIT(ADC0MOD0,            0x4002D044,__READ_WRITE ,__adcmod0_bits);
__IO_REG32_BIT(ADC0MOD1,            0x4002D048,__READ_WRITE ,__adcmod1_bits);
__IO_REG32_BIT(ADC0ENA,             0x4002D04C,__READ_WRITE ,__adcena_bits);
__IO_REG32_BIT(ADC0FLG,             0x4002D050,__READ       ,__adcflg_bits);

/***************************************************************************
 **
 ** ADC1
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC1RSLT0,           0x4002F000,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT1,           0x4002F004,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT2,           0x4002F008,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT3,           0x4002F00C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT4,           0x4002F010,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT5,           0x4002F014,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT6,           0x4002F018,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT7,           0x4002F01C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT8,           0x4002F020,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT9,           0x4002F024,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT10,          0x4002F028,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT11,          0x4002F02C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT12,          0x4002F030,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1RSLT13,          0x4002F034,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC1SETI0,          	0x4002F038,__READ_WRITE ,__adcseti0_bits);
__IO_REG32_BIT(ADC1SETI1,         	0x4002F03C,__READ_WRITE ,__adcseti1_bits);
__IO_REG32_BIT(ADC1SETT,         		0x4002F040,__READ_WRITE ,__adcsett_bits);
__IO_REG32_BIT(ADC1MOD0,            0x4002F044,__READ_WRITE ,__adcmod0_bits);
__IO_REG32_BIT(ADC1MOD1,            0x4002F048,__READ_WRITE ,__adcmod1_bits);
__IO_REG32_BIT(ADC1ENA,             0x4002F04C,__READ_WRITE ,__adcena_bits);
__IO_REG32_BIT(ADC1FLG,             0x4002F050,__READ       ,__adcflg_bits);

/***************************************************************************
 **
 ** ADC2
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC2RSLT0,           0x4001E000,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT1,           0x4001E004,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT2,           0x4001E008,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT3,           0x4001E00C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT4,           0x4001E010,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT5,           0x4001E014,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT6,           0x4001E018,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT7,           0x4001E01C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT8,           0x4001E020,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT9,           0x4001E024,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT10,          0x4001E028,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT11,          0x4001E02C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT12,          0x4001E030,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2RSLT13,          0x4001E034,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC2SETI0,          	0x4001E038,__READ_WRITE ,__adcseti0_bits);
__IO_REG32_BIT(ADC2SETI1,         	0x4001E03C,__READ_WRITE ,__adcseti1_bits);
__IO_REG32_BIT(ADC2SETT,         		0x4001E040,__READ_WRITE ,__adcsett_bits);
__IO_REG32_BIT(ADC2MOD0,            0x4001E044,__READ_WRITE ,__adcmod0_bits);
__IO_REG32_BIT(ADC2MOD1,            0x4001E048,__READ_WRITE ,__adcmod1_bits);
__IO_REG32_BIT(ADC2ENA,             0x4001E04C,__READ_WRITE ,__adcena_bits);
__IO_REG32_BIT(ADC2FLG,             0x4001E050,__READ       ,__adcflg_bits);

/***************************************************************************
 **
 ** ADC3
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC3RSLT0,           0x4001F000,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT1,           0x4001F004,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT2,           0x4001F008,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT3,           0x4001F00C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT4,           0x4001F010,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT5,           0x4001F014,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT6,           0x4001F018,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT7,           0x4001F01C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT8,           0x4001F020,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT9,           0x4001F024,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT10,          0x4001F028,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT11,          0x4001F02C,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT12,          0x4001F030,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3RSLT13,          0x4001F034,__READ       ,__adcrslt_bits);
__IO_REG32_BIT(ADC3SETI0,          	0x4001F038,__READ_WRITE ,__adcseti0_bits);
__IO_REG32_BIT(ADC3SETI1,         	0x4001F03C,__READ_WRITE ,__adcseti1_bits);
__IO_REG32_BIT(ADC3SETT,         		0x4001F040,__READ_WRITE ,__adcsett_bits);
__IO_REG32_BIT(ADC3MOD0,            0x4001F044,__READ_WRITE ,__adcmod0_bits);
__IO_REG32_BIT(ADC3MOD1,            0x4001F048,__READ_WRITE ,__adcmod1_bits);
__IO_REG32_BIT(ADC3ENA,             0x4001F04C,__READ_WRITE ,__adcena_bits);
__IO_REG32_BIT(ADC3FLG,             0x4001F050,__READ       ,__adcflg_bits);

/***************************************************************************
 **
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(FCSECBIT,            0x41FFF010, __READ_WRITE , __secbit_bits);
__IO_REG32_BIT(FCFLCSR0,            0x41FFF020, __READ       , __flcsr0_bits);
__IO_REG32_BIT(FCFLCSR1,            0x41FFF030, __READ       , __flcsr1_bits);
__IO_REG32_BIT(FCFLSR0,             0x41FFF400, __READ       , __flsr0_bits);
__IO_REG32_BIT(FCFLSR1,             0x41FFF404, __READ       , __flsr1_bits);
__IO_REG32_BIT(FCFLCR0,             0x41FFF40C, __READ_WRITE , __flcr0_bits);
__IO_REG32_BIT(FCOVLADR0,           0x41FFF410, __READ_WRITE , __ovladr_bits);
__IO_REG32_BIT(FCOVLADR1,           0x41FFF414, __READ_WRITE , __ovladr_bits);
__IO_REG32_BIT(FCOVLADR2,           0x41FFF418, __READ_WRITE , __ovladr_bits);
__IO_REG32_BIT(FCOVLADR3,           0x41FFF41C, __READ_WRITE , __ovladr_bits);
__IO_REG32_BIT(FCOVLEN,             0x41FFF420, __READ_WRITE , __ovlen_bits);
__IO_REG32_BIT(FCOVLMOD,            0x41FFF424, __READ_WRITE , __ovlmod_bits);

/***************************************************************************
 **
 ** fRNET
 **
 ***************************************************************************/
__IO_REG32(    fWDATA,              0xE00FEF00, __WRITE      );
__IO_REG32_BIT(fRDATA,              0xE00FEF04, __READ       , __frdata_bits);

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  TMPM354F10TFG DMA0 Lines
 **
 ***************************************************************************/
#define DMA0_ADC1    			  0
#define DMA0_ADC2    			  1
#define DMA0_ADC3    			  2
#define DMA0_CI1ZT0   		  3
#define DMA0_CI1PT0   		  4
#define DMA0_CI2PT0    		  5
#define DMA0_CI2QT0   		  6
#define DMA0_CI4T0    		  7
#define DMA0_CI8T0    		  8
#define DMA0_CIUT0    		  9
#define DMA0_CI1ZT1				  10
#define DMA0_CI1PT1				  11
#define DMA0_CI2PT1				  12
#define DMA0_CI2QT1				  13
#define DMA0_CI1ZT2				  14
#define DMA0_CI1PT2				  15
#define DMA0_CI2PT2 			  16
#define DMA0_CI2QT2  			  17
#define DMA0_CPWM0      		18
#define DMA0_CPWM1      		19
#define DMA0_FPWM0      		20
#define DMA0_FPWM1      		21
#define DMA0_ESEI0RX    		22
#define DMA0_ESEI0TX      	23
#define DMA0_ESEI1RX    		24
#define DMA0_ESEI1TX      	25
#define DMA0_RX0    			  26
#define DMA0_TX0    			  27
#define DMA0_RX1    			  28
#define DMA0_TX1    			  29
#define DMA0_RX2	  			  30
#define DMA0_TX2	  			  31

/***************************************************************************
 **
 **  TMPM354F10TFG DMA1 Lines
 **
 ***************************************************************************/
 
#define DMA1_ADC0 			    0
#define DMA1_CI1ZT0 			  1
#define DMA1_RSINZ  			  2
#define DMA1_RCOSZ  		    3
#define DMA1_TCAP0R  		    4
#define DMA1_TCAP0F   		  5
#define DMA1_TCAP1R  		    6
#define DMA1_TCAP1F  		    7
#define DMA1_TCAP2R  		    8
#define DMA1_TCAP2F  		    9
#define DMA1_TCAP3R			    10
#define DMA1_TCAP3F			    11
#define DMA1_TCAP4R			    12
#define DMA1_TCAP4F			    13
#define DMA1_TCAP5R			    14
#define DMA1_TCAP5F			    15
#define DMA1_TCMP0			    16
#define DMA1_TCMP1 			    17
#define DMA1_CMP00    		  18
#define DMA1_CMP01    		  19
#define DMA1_CMP10    		  20
#define DMA1_CMP11    		  21
#define DMA1_CMP20    		  22
#define DMA1_CMP21        	23
#define DMA1_CMP30    	  	24
#define DMA1_CMP31        	25
#define DMA1_CMP40			    26
#define DMA1_CMP41			    27
#define DMA1_PWM0			      28
#define DMA1_PWM1			      29
#define DMA1_PWM2			      30
#define DMA1_PWM3			      31

/***************************************************************************
 **
 **  TMPM354F10TFG Interrupt Lines
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
#define INTWARN               17
#define INTINFO               18
#define INTWDTERR             19
#define INTPOEMG              20
#define INTCOEMG              21
#define INTOVF0               22
#define INTCMP00              23
#define INTCMP01              24
#define INTOVF1               25
#define INTCMP10              26
#define INTCMP11              27
#define INTOVF2               28
#define INTCMP20              29
#define INTCMP21              30
#define INTOVF3               31
#define INTCMP30              32
#define INTCMP31              33
#define INTOVF4               34
#define INTCMP40              35
#define INTCMP41              36
#define INTTBTOVF             37
#define INTTBTI0              38
#define INTTBTI1              39
#define INTTBTI2              40
#define INTTBTI3              41
#define INTTCAP0R             42
#define INTTCAP0F             43
#define INTTCAP1R             44
#define INTTCAP1F             45
#define INTTCAP2R             46
#define INTTCAP2F             47
#define INTTCAP3R             48
#define INTTCAP3F             49
#define INTTCAP4R             50
#define INTTCAP4F             51
#define INTTCAP5R             52
#define INTTCAP5F             53
#define INTTCMP0              54
#define INTTCMP1              55
#define INTTCAP7R             56
#define INTTCAP7F             57
#define INTPWM0               58
#define INTPWM1               59
#define INTPWM2               60
#define INTPWM3               61
#define INTRX0                62
#define INTTX0                63
#define INTRX1                64
#define INTTX1                65
#define INTRX2                66
#define INTTX2                67
#define INTESEIERR0           68
#define INTESEIRX0            69
#define INTESEITX0            70
#define INTESEIERR1           71
#define INTESEIRX1            72
#define INTESEITX1            73
#define INTCR0                74
#define INTCT0                75
#define INTCG0                76
#define INTCR1                77
#define INTCT1                78
#define INTCG1                79
#define INTCR2                80
#define INTCT2                81
#define INTCG2                82
#define INTADC0               83
#define INTADC1               84
#define INTADC2               85
#define INTADC3               86
#define INTEXC                87
#define INTVE                 88
#define INTRDC0               89
#define INTZ                  90
#define INTCIUT0              91
#define INTCAR0A              92
#define INTCAR0B              93
#define INTCAR1               94
#define INTCAR2               95
#define INTCPWM0              96
#define INTCPWM1              97
#define INTFPWM0              98
#define INTFPWM1              99
#define INTDMAXFEND0          100
#define INTDMAXFEND1          101
#define INTTCAP13R            102
#define INTRDC1               103
#define INTLVCLR              104

#endif    /* __IOTMPM354F10TFG_H */

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
Interrupt10  = INTWARN        0x44
Interrupt11  = INTINFO        0x48
Interrupt12  = INTWDTERR      0x4C
Interrupt13  = INTPOEMG       0x50
Interrupt14  = INTCOEMG       0x54
Interrupt15  = INTOVF0        0x58
Interrupt16  = INTCMP00       0x5C
Interrupt17  = INTCMP01       0x60
Interrupt18  = INTOVF1        0x64
Interrupt19  = INTCMP10       0x68
Interrupt20  = INTCMP11       0x6C
Interrupt21  = INTOVF2        0x70
Interrupt22  = INTCMP20       0x74
Interrupt23  = INTCMP21       0x78
Interrupt24  = INTOVF3        0x7C
Interrupt25  = INTCMP30       0x80
Interrupt26  = INTCMP31       0x84
Interrupt27  = INTOVF4        0x88
Interrupt28  = INTCMP40       0x8C
Interrupt29  = INTCMP41       0x90
Interrupt30  = INTTBTOVF      0x94
Interrupt31  = INTTBTI0       0x98
Interrupt32  = INTTBTI1       0x9C
Interrupt33  = INTTBTI2       0xA0
Interrupt34  = INTTBTI3       0xA4
Interrupt35  = INTTCAP0R      0xA8
Interrupt36  = INTTCAP0F      0xAC
Interrupt37  = INTTCAP1R      0xB0
Interrupt38  = INTTCAP1F      0xB4
Interrupt39  = INTTCAP2R      0xB8
Interrupt40  = INTTCAP2F      0xBC
Interrupt41  = INTTCAP3R      0xC0
Interrupt42  = INTTCAP3F      0xC4
Interrupt43  = INTTCAP4R      0xC8
Interrupt44  = INTTCAP4F      0xCC
Interrupt45  = INTTCAP5R      0xD0
Interrupt46  = INTTCAP5F      0xD4
Interrupt47  = INTTCMP0       0xD8
Interrupt48  = INTTCMP1       0xDC
Interrupt49  = INTTCAP7R      0xE0
Interrupt50  = INTTCAP7F      0xE4
Interrupt51  = INTPWM0        0xE8
Interrupt52  = INTPWM1        0xEC
Interrupt53  = INTPWM2        0xF0
Interrupt54  = INTPWM3        0xF4
Interrupt55  = INTRX0         0xF8
Interrupt56  = INTTX0         0xFC
Interrupt57  = INTRX1         0x100
Interrupt58  = INTTX1         0x104
Interrupt59  = INTRX2         0x108
Interrupt60  = INTTX2         0x10C
Interrupt61  = INTESEIERR0    0x110
Interrupt62  = INTESEIRX0     0x114
Interrupt63  = INTESEITX0     0x118
Interrupt64  = INTESEIERR1    0x11C
Interrupt65  = INTESEIRX1     0x120
Interrupt66  = INTESEITX1     0x124
Interrupt67  = INTCR0         0x128
Interrupt68  = INTCT0         0x12C
Interrupt69  = INTCG0         0x130
Interrupt70  = INTCR1         0x134
Interrupt71  = INTCT1         0x138
Interrupt72  = INTCG1         0x13C
Interrupt73  = INTCR2         0x140
Interrupt74  = INTCT2         0x144
Interrupt75  = INTCG2         0x148
Interrupt76  = INTADC0        0x14C
Interrupt77  = INTADC1        0x150
Interrupt78  = INTADC2        0x154
Interrupt79  = INTADC3        0x158
Interrupt80  = INTEXC         0x15C
Interrupt81  = INTVE          0x160
Interrupt82  = INTRDC0        0x164
Interrupt83  = INTZ           0x168
Interrupt84  = INTCIUT0       0x16C
Interrupt85  = INTCAR0A       0x170
Interrupt86  = INTCAR0B       0x174
Interrupt87  = INTCAR1        0x178
Interrupt88  = INTCAR2        0x17C
Interrupt89  = INTCPWM0       0x180
Interrupt90  = INTCPWM1       0x184
Interrupt91  = INTFPWM0       0x188
Interrupt92  = INTFPWM1       0x18C
Interrupt93  = INTDMAXFEND0   0x190
Interrupt94  = INTDMAXFEND1   0x194
Interrupt95  = INTTCAP13R     0x198
Interrupt96  = INTRDC1        0x19C
Interrupt97  = INTLVCLR       0x1A0
###DDF-INTERRUPT-END###*/