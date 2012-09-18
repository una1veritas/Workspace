/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    ST SPEAR310
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2010
 **
 **    $Revision: 40530 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOSPEAR310_H
#define __IOSPEAR310_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x58 = 88 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   SPEAR310 SPECIAL FUNCTION REGISTERS
 **
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

/* VICIRQSTATUS register */
typedef struct
{
  __REG32  IRQStatus0     : 1;
  __REG32  IRQStatus1     : 1;
  __REG32  IRQStatus2     : 1;
  __REG32  IRQStatus3     : 1;
  __REG32  IRQStatus4     : 1;
  __REG32  IRQStatus5     : 1;
  __REG32  IRQStatus6     : 1;
  __REG32  IRQStatus7     : 1;
  __REG32  IRQStatus8     : 1;
  __REG32  IRQStatus9     : 1;
  __REG32  IRQStatus10    : 1;
  __REG32  IRQStatus11    : 1;
  __REG32  IRQStatus12    : 1;
  __REG32  IRQStatus13    : 1;
  __REG32  IRQStatus14    : 1;
  __REG32  IRQStatus15    : 1;
  __REG32  IRQStatus16    : 1;
  __REG32  IRQStatus17    : 1;
  __REG32  IRQStatus18    : 1;
  __REG32  IRQStatus19    : 1;
  __REG32  IRQStatus20    : 1;
  __REG32  IRQStatus21    : 1;
  __REG32  IRQStatus22    : 1;
  __REG32  IRQStatus23    : 1;
  __REG32  IRQStatus24    : 1;
  __REG32  IRQStatus25    : 1;
  __REG32  IRQStatus26    : 1;
  __REG32  IRQStatus27    : 1;
  __REG32  IRQStatus28    : 1;
  __REG32  IRQStatus29    : 1;
  __REG32  IRQStatus30    : 1;
  __REG32  IRQStatus31    : 1;
} __vicirqstatus_bits;

/* VICFIQSTATUS register */
typedef struct
{
  __REG32  FIQStatus0     : 1;
  __REG32  FIQStatus1     : 1;
  __REG32  FIQStatus2     : 1;
  __REG32  FIQStatus3     : 1;
  __REG32  FIQStatus4     : 1;
  __REG32  FIQStatus5     : 1;
  __REG32  FIQStatus6     : 1;
  __REG32  FIQStatus7     : 1;
  __REG32  FIQStatus8     : 1;
  __REG32  FIQStatus9     : 1;
  __REG32  FIQStatus10    : 1;
  __REG32  FIQStatus11    : 1;
  __REG32  FIQStatus12    : 1;
  __REG32  FIQStatus13    : 1;
  __REG32  FIQStatus14    : 1;
  __REG32  FIQStatus15    : 1;
  __REG32  FIQStatus16    : 1;
  __REG32  FIQStatus17    : 1;
  __REG32  FIQStatus18    : 1;
  __REG32  FIQStatus19    : 1;
  __REG32  FIQStatus20    : 1;
  __REG32  FIQStatus21    : 1;
  __REG32  FIQStatus22    : 1;
  __REG32  FIQStatus23    : 1;
  __REG32  FIQStatus24    : 1;
  __REG32  FIQStatus25    : 1;
  __REG32  FIQStatus26    : 1;
  __REG32  FIQStatus27    : 1;
  __REG32  FIQStatus28    : 1;
  __REG32  FIQStatus29    : 1;
  __REG32  FIQStatus30    : 1;
  __REG32  FIQStatus31    : 1;
} __vicfiqstatus_bits;

/* VICRAWINTR register */
typedef struct
{
  __REG32  RawStatus0     : 1;
  __REG32  RawStatus1     : 1;
  __REG32  RawStatus2     : 1;
  __REG32  RawStatus3     : 1;
  __REG32  RawStatus4     : 1;
  __REG32  RawStatus5     : 1;
  __REG32  RawStatus6     : 1;
  __REG32  RawStatus7     : 1;
  __REG32  RawStatus8     : 1;
  __REG32  RawStatus9     : 1;
  __REG32  RawStatus10    : 1;
  __REG32  RawStatus11    : 1;
  __REG32  RawStatus12    : 1;
  __REG32  RawStatus13    : 1;
  __REG32  RawStatus14    : 1;
  __REG32  RawStatus15    : 1;
  __REG32  RawStatus16    : 1;
  __REG32  RawStatus17    : 1;
  __REG32  RawStatus18    : 1;
  __REG32  RawStatus19    : 1;
  __REG32  RawStatus20    : 1;
  __REG32  RawStatus21    : 1;
  __REG32  RawStatus22    : 1;
  __REG32  RawStatus23    : 1;
  __REG32  RawStatus24    : 1;
  __REG32  RawStatus25    : 1;
  __REG32  RawStatus26    : 1;
  __REG32  RawStatus27    : 1;
  __REG32  RawStatus28    : 1;
  __REG32  RawStatus29    : 1;
  __REG32  RawStatus30    : 1;
  __REG32  RawStatus31    : 1;
} __vicrawintr_bits;

/* VICINTSELECT register */
typedef struct
{
  __REG32  IntSelect0     : 1;
  __REG32  IntSelect1     : 1;
  __REG32  IntSelect2     : 1;
  __REG32  IntSelect3     : 1;
  __REG32  IntSelect4     : 1;
  __REG32  IntSelect5     : 1;
  __REG32  IntSelect6     : 1;
  __REG32  IntSelect7     : 1;
  __REG32  IntSelect8     : 1;
  __REG32  IntSelect9     : 1;
  __REG32  IntSelect10    : 1;
  __REG32  IntSelect11    : 1;
  __REG32  IntSelect12    : 1;
  __REG32  IntSelect13    : 1;
  __REG32  IntSelect14    : 1;
  __REG32  IntSelect15    : 1;
  __REG32  IntSelect16    : 1;
  __REG32  IntSelect17    : 1;
  __REG32  IntSelect18    : 1;
  __REG32  IntSelect19    : 1;
  __REG32  IntSelect20    : 1;
  __REG32  IntSelect21    : 1;
  __REG32  IntSelect22    : 1;
  __REG32  IntSelect23    : 1;
  __REG32  IntSelect24    : 1;
  __REG32  IntSelect25    : 1;
  __REG32  IntSelect26    : 1;
  __REG32  IntSelect27    : 1;
  __REG32  IntSelect28    : 1;
  __REG32  IntSelect29    : 1;
  __REG32  IntSelect30    : 1;
  __REG32  IntSelect31    : 1;
} __vicintselect_bits;

/* VICINTENABLE register */
typedef struct
{
  __REG32  IntEnable0     : 1;
  __REG32  IntEnable1     : 1;
  __REG32  IntEnable2     : 1;
  __REG32  IntEnable3     : 1;
  __REG32  IntEnable4     : 1;
  __REG32  IntEnable5     : 1;
  __REG32  IntEnable6     : 1;
  __REG32  IntEnable7     : 1;
  __REG32  IntEnable8     : 1;
  __REG32  IntEnable9     : 1;
  __REG32  IntEnable10    : 1;
  __REG32  IntEnable11    : 1;
  __REG32  IntEnable12    : 1;
  __REG32  IntEnable13    : 1;
  __REG32  IntEnable14    : 1;
  __REG32  IntEnable15    : 1;
  __REG32  IntEnable16    : 1;
  __REG32  IntEnable17    : 1;
  __REG32  IntEnable18    : 1;
  __REG32  IntEnable19    : 1;
  __REG32  IntEnable20    : 1;
  __REG32  IntEnable21    : 1;
  __REG32  IntEnable22    : 1;
  __REG32  IntEnable23    : 1;
  __REG32  IntEnable24    : 1;
  __REG32  IntEnable25    : 1;
  __REG32  IntEnable26    : 1;
  __REG32  IntEnable27    : 1;
  __REG32  IntEnable28    : 1;
  __REG32  IntEnable29    : 1;
  __REG32  IntEnable30    : 1;
  __REG32  IntEnable31    : 1;
} __vicintenable_bits;

/* VICINTENCLEAR register */
typedef struct
{
  __REG32  IntEnableClear0     : 1;
  __REG32  IntEnableClear1     : 1;
  __REG32  IntEnableClear2     : 1;
  __REG32  IntEnableClear3     : 1;
  __REG32  IntEnableClear4     : 1;
  __REG32  IntEnableClear5     : 1;
  __REG32  IntEnableClear6     : 1;
  __REG32  IntEnableClear7     : 1;
  __REG32  IntEnableClear8     : 1;
  __REG32  IntEnableClear9     : 1;
  __REG32  IntEnableClear10    : 1;
  __REG32  IntEnableClear11    : 1;
  __REG32  IntEnableClear12    : 1;
  __REG32  IntEnableClear13    : 1;
  __REG32  IntEnableClear14    : 1;
  __REG32  IntEnableClear15    : 1;
  __REG32  IntEnableClear16    : 1;
  __REG32  IntEnableClear17    : 1;
  __REG32  IntEnableClear18    : 1;
  __REG32  IntEnableClear19    : 1;
  __REG32  IntEnableClear20    : 1;
  __REG32  IntEnableClear21    : 1;
  __REG32  IntEnableClear22    : 1;
  __REG32  IntEnableClear23    : 1;
  __REG32  IntEnableClear24    : 1;
  __REG32  IntEnableClear25    : 1;
  __REG32  IntEnableClear26    : 1;
  __REG32  IntEnableClear27    : 1;
  __REG32  IntEnableClear28    : 1;
  __REG32  IntEnableClear29    : 1;
  __REG32  IntEnableClear30    : 1;
  __REG32  IntEnableClear31    : 1;
} __vicintenclear_bits;

/* VICSOFTINT register */
typedef struct
{
  __REG32  SoftInt0     : 1;
  __REG32  SoftInt1     : 1;
  __REG32  SoftInt2     : 1;
  __REG32  SoftInt3     : 1;
  __REG32  SoftInt4     : 1;
  __REG32  SoftInt5     : 1;
  __REG32  SoftInt6     : 1;
  __REG32  SoftInt7     : 1;
  __REG32  SoftInt8     : 1;
  __REG32  SoftInt9     : 1;
  __REG32  SoftInt10    : 1;
  __REG32  SoftInt11    : 1;
  __REG32  SoftInt12    : 1;
  __REG32  SoftInt13    : 1;
  __REG32  SoftInt14    : 1;
  __REG32  SoftInt15    : 1;
  __REG32  SoftInt16    : 1;
  __REG32  SoftInt17    : 1;
  __REG32  SoftInt18    : 1;
  __REG32  SoftInt19    : 1;
  __REG32  SoftInt20    : 1;
  __REG32  SoftInt21    : 1;
  __REG32  SoftInt22    : 1;
  __REG32  SoftInt23    : 1;
  __REG32  SoftInt24    : 1;
  __REG32  SoftInt25    : 1;
  __REG32  SoftInt26    : 1;
  __REG32  SoftInt27    : 1;
  __REG32  SoftInt28    : 1;
  __REG32  SoftInt29    : 1;
  __REG32  SoftInt30    : 1;
  __REG32  SoftInt31    : 1;
} __vicsoftint_bits;

/* VICSOFTINTCLEAR register */
typedef struct
{
  __REG32  SoftIntClear0     : 1;
  __REG32  SoftIntClear1     : 1;
  __REG32  SoftIntClear2     : 1;
  __REG32  SoftIntClear3     : 1;
  __REG32  SoftIntClear4     : 1;
  __REG32  SoftIntClear5     : 1;
  __REG32  SoftIntClear6     : 1;
  __REG32  SoftIntClear7     : 1;
  __REG32  SoftIntClear8     : 1;
  __REG32  SoftIntClear9     : 1;
  __REG32  SoftIntClear10    : 1;
  __REG32  SoftIntClear11    : 1;
  __REG32  SoftIntClear12    : 1;
  __REG32  SoftIntClear13    : 1;
  __REG32  SoftIntClear14    : 1;
  __REG32  SoftIntClear15    : 1;
  __REG32  SoftIntClear16    : 1;
  __REG32  SoftIntClear17    : 1;
  __REG32  SoftIntClear18    : 1;
  __REG32  SoftIntClear19    : 1;
  __REG32  SoftIntClear20    : 1;
  __REG32  SoftIntClear21    : 1;
  __REG32  SoftIntClear22    : 1;
  __REG32  SoftIntClear23    : 1;
  __REG32  SoftIntClear24    : 1;
  __REG32  SoftIntClear25    : 1;
  __REG32  SoftIntClear26    : 1;
  __REG32  SoftIntClear27    : 1;
  __REG32  SoftIntClear28    : 1;
  __REG32  SoftIntClear29    : 1;
  __REG32  SoftIntClear30    : 1;
  __REG32  SoftIntClear31    : 1;
} __vicsoftintclear_bits;

/* VICPROTECTION register */
typedef struct
{
  __REG32  Protection        : 1;
  __REG32                    :31;
} __vicprotection_bits;

/* VICVECTCNTL register */
typedef struct
{
  __REG32  IntSource         : 5;
  __REG32  E                 : 1;
  __REG32                    :26;
} __vicvectcntl_bits;

/* VICPERIPHID1 register */
typedef struct
{
  __REG8   Partnumber        : 4;
  __REG8   Designer          : 4;
} __vicperiphid1_bits;

/* VICPERIPHID2 register */
typedef struct
{
  __REG8   Designer          : 4;
  __REG8   Revision          : 4;
} __vicperiphid2_bits;

/* VICPERIPHID3 register */
typedef struct
{
  __REG8   Designer          : 4;
  __REG8   Revision          : 4;
} __vicperiphid3_bits;

/* MEM0_CTL register */
typedef struct
{
  __REG32  ADDR_CMP_EN       : 4;
  __REG32                    : 4;
  __REG32  AHB0_FIFO_TYPE    : 4;
  __REG32                    : 4;
  __REG32  AHB1_FIFO_TYPE    : 4;
  __REG32                    : 4;
  __REG32  AHB2_FIFO_TYPE    : 4;
  __REG32                    : 4;
} __mem0_ctl_bits;

/* MEM1_CTL register */
typedef struct
{
  __REG32  AHB3_FIFO_TYPE    : 2;
  __REG32                    : 6;
  __REG32  AHB4_FIFO_TYPE    : 2;
  __REG32                    :22;
} __mem1_ctl_bits;

/* MEM2_CTL register */
typedef struct
{
  __REG32  AP                : 1;
  __REG32                    : 7;
  __REG32  AREFRESH          : 1;
  __REG32                    : 7;
  __REG32  AUTO_RFSH_MODE    : 1;
  __REG32                    : 7;
  __REG32  BANK_SPLI_EN      : 1;
  __REG32                    : 7;
} __mem2_ctl_bits;

/* MEM3_CTL register */
typedef struct
{
  __REG32  CONCURRENTAP      : 1;
  __REG32                    : 7;
  __REG32  DDRII_DDRI_MODE   : 1;
  __REG32                    : 7;
  __REG32  DLLLOCK           : 1;
  __REG32                    : 7;
  __REG32  DLL_BYPASS_MODE   : 1;
  __REG32                    : 7;
} __mem3_ctl_bits;

/* MEM4_CTL register */
typedef struct
{
  __REG32  DQS_N_EN          : 1;
  __REG32                    : 7;
  __REG32  EIGHT_BANK_MODE   : 1;
  __REG32                    : 7;
  __REG32  FAST_WRITE        : 1;
  __REG32                    : 7;
  __REG32  INTRPTAPBURST     : 1;
  __REG32                    : 7;
} __mem4_ctl_bits;

/* MEM5_CTL register */
typedef struct
{
  __REG32  INTRPTREADA          : 1;
  __REG32                       : 7;
  __REG32  INTRPTWRITEA         : 1;
  __REG32                       : 7;
  __REG32  NOCMDINIT            : 1;
  __REG32                       : 7;
  __REG32  ODT_ADD_TURN_CLK_EN  : 1;
  __REG32                       : 7;
} __mem5_ctl_bits;

/* MEM6_CTL register */
typedef struct
{
  __REG32  PLACEMENT_EN         : 1;
  __REG32                       : 7;
  __REG32  POWER_DOWN           : 1;
  __REG32                       : 7;
  __REG32  PRIORITY_EN          : 1;
  __REG32                       : 7;
  __REG32  REDUC                : 1;
  __REG32                       : 7;
} __mem6_ctl_bits;

/* MEM7_CTL register */
typedef struct
{
  __REG32  REG_DIMM_EN          : 1;
  __REG32                       : 7;
  __REG32  SW_SAME_EN           : 1;
  __REG32                       : 7;
  __REG32  SREFRESH             : 1;
  __REG32                       : 7;
  __REG32  START                : 1;
  __REG32                       : 7;
} __mem7_ctl_bits;

/* MEM8_CTL register */
typedef struct
{
  __REG32  TRAS_LOCKOUT         : 1;
  __REG32                       : 7;
  __REG32  WRRLC                : 1;
  __REG32                       : 7;
  __REG32  WRITE_INTRPT         : 1;
  __REG32                       : 7;
  __REG32  WRITE_MODE_REG       : 1;
  __REG32                       : 7;
} __mem8_ctl_bits;

/* MEM9_CTL register */
typedef struct
{
  __REG32  CS_MAP               : 2;
  __REG32                       : 6;
  __REG32  MAX_CS               : 2;
  __REG32                       : 6;
  __REG32  ODT_RD_MAP_CS0       : 2;
  __REG32                       : 6;
  __REG32  ODT_RD_MAP_CS1       : 2;
  __REG32                       : 6;
} __mem9_ctl_bits;

/* MEM10_CTL register */
typedef struct
{
  __REG32  ODT_WR_MAP_CS0       : 2;
  __REG32                       : 6;
  __REG32  ODT_WR_MAP_CS1       : 2;
  __REG32                       : 6;
  __REG32  OUTOFRANGETYPE       : 2;
  __REG32                       : 6;
  __REG32  RTT_0                : 2;
  __REG32                       : 6;
} __mem10_ctl_bits;

/* MEM11_CTL register */
typedef struct
{
  __REG32  RTT_PAD_TERMINATION  : 2;
  __REG32                       : 6;
  __REG32  ADDR_PINS            : 3;
  __REG32                       : 5;
  __REG32  AHB0_PORT_ORDERING   : 3;
  __REG32                       : 5;
  __REG32  AHB0_R_PRIOTRITY     : 3;
  __REG32                       : 5;
} __mem11_ctl_bits;

/* MEM12_CTL register */
typedef struct
{
  __REG32  AHB0_W_PRIORITY      : 3;
  __REG32                       : 5;
  __REG32  AHB1_PORT_ORDERING   : 3;
  __REG32                       : 5;
  __REG32  AHB1_R_PRIORITY      : 3;
  __REG32                       : 5;
  __REG32  AHB1_W_PRIORITY      : 3;
  __REG32                       : 5;
} __mem12_ctl_bits;

/* MEM13_CTL register */
typedef struct
{
  __REG32  AHB2_PORT_ORDERING   : 3;
  __REG32                       : 5;
  __REG32  AHB2_R_PRIORITY      : 3;
  __REG32                       : 5;
  __REG32  AHB2_W_PRIORITY      : 3;
  __REG32                       : 5;
  __REG32  AHB3_PORT_ORDERING   : 3;
  __REG32                       : 5;
} __mem13_ctl_bits;

/* MEM14_CTL register */
typedef struct
{
  __REG32  AHB3_R_PRIORITY      : 3;
  __REG32                       : 5;
  __REG32  AHB3_W_PRIORITY      : 3;
  __REG32                       : 5;
  __REG32  AHB4_PORT_ORDERING   : 3;
  __REG32                       : 5;
  __REG32  AHB4_R_PRIORITY      : 3;
  __REG32                       : 5;
} __mem14_ctl_bits;

/* MEM15_CTL register */
typedef struct
{
  __REG32  AHB4_W_PRIORITY      : 3;
  __REG32                       :29;
} __mem15_ctl_bits;

/* MEM17_CTL register */
typedef struct
{
  __REG32  CAS_LATENCY            : 3;
  __REG32                         : 5;
  __REG32  COLUMN_SIZE            : 3;
  __REG32                         : 5;
  __REG32  OUT_OF_RANGE_SOURCE_ID : 3;
  __REG32                         : 5;
  __REG32  TCKE                   : 3;
  __REG32                         : 5;
} __mem17_ctl_bits;

/* MEM18_CTL register */
typedef struct
{
  __REG32  TEMRS                  : 3;
  __REG32                         :13;
  __REG32  TRRD                   : 3;
  __REG32                         : 5;
  __REG32  TRTP                   : 3;
  __REG32                         : 5;
} __mem18_ctl_bits;

/* MEM19_CTL register */
typedef struct
{
  __REG32  TWR_INT                : 3;
  __REG32                         : 5;
  __REG32  TWTR                   : 3;
  __REG32                         : 5;
  __REG32  WRRWS                  : 2;
  __REG32                         : 6;
  __REG32  WRLAT                  : 3;
  __REG32                         : 5;
} __mem19_ctl_bits;

/* MEM20_CTL register */
typedef struct
{
  __REG32  AGE_COUNT              : 6;
  __REG32                         : 2;
  __REG32  AHB0_P0RP              : 4;
  __REG32                         : 4;
  __REG32  AHB0_P1RP              : 4;
  __REG32                         : 4;
  __REG32  AHB0_P2RP              : 4;
  __REG32                         : 4;
} __mem20_ctl_bits;

/* MEM21_CTL register */
typedef struct
{
  __REG32  AHB0_P3RP              : 4;
  __REG32                         : 4;
  __REG32  AHB0_P4RP              : 4;
  __REG32                         : 4;
  __REG32  AHB0_P5RP              : 4;
  __REG32                         : 4;
  __REG32  AHB0_P6RP              : 4;
  __REG32                         : 4;
} __mem21_ctl_bits;

/* MEM22_CTL register */
typedef struct
{
  __REG32  AHB0_P7RP              : 4;
  __REG32                         : 4;
  __REG32  AHB1_P0RP              : 4;
  __REG32                         : 4;
  __REG32  AHB1_P1RP              : 4;
  __REG32                         : 4;
  __REG32  AHB1_P2RP              : 4;
  __REG32                         : 4;
} __mem22_ctl_bits;

/* MEM23_CTL register */
typedef struct
{
  __REG32  AHB1_P3RP              : 4;
  __REG32                         : 4;
  __REG32  AHB1_P4RP              : 4;
  __REG32                         : 4;
  __REG32  AHB1_P5RP              : 4;
  __REG32                         : 4;
  __REG32  AHB1_P6RP              : 4;
  __REG32                         : 4;
} __mem23_ctl_bits;

/* MEM24_CTL register */
typedef struct
{
  __REG32  AHB1_P7RP              : 4;
  __REG32                         : 4;
  __REG32  AHB2_P0RP              : 4;
  __REG32                         : 4;
  __REG32  AHB2_P1RP              : 4;
  __REG32                         : 4;
  __REG32  AHB2_P2RP              : 4;
  __REG32                         : 4;
} __mem24_ctl_bits;

/* MEM25_CTL register */
typedef struct
{
  __REG32  AHB2_P3RP              : 4;
  __REG32                         : 4;
  __REG32  AHB2_P4RP              : 4;
  __REG32                         : 4;
  __REG32  AHB2_P5RP              : 4;
  __REG32                         : 4;
  __REG32  AHB2_P6RP              : 4;
  __REG32                         : 4;
} __mem25_ctl_bits;

/* MEM26_CTL register */
typedef struct
{
  __REG32  AHB2_P7RP              : 4;
  __REG32                         : 4;
  __REG32  AHB3_P0RP              : 4;
  __REG32                         : 4;
  __REG32  AHB3_P1RP              : 4;
  __REG32                         : 4;
  __REG32  AHB3_P2RP              : 4;
  __REG32                         : 4;
} __mem26_ctl_bits;

/* MEM27_CTL register */
typedef struct
{
  __REG32  AHB3_P3RP              : 4;
  __REG32                         : 4;
  __REG32  AHB3_P4RP              : 4;
  __REG32                         : 4;
  __REG32  AHB3_P5RP              : 4;
  __REG32                         : 4;
  __REG32  AHB3_P6RP              : 4;
  __REG32                         : 4;
} __mem27_ctl_bits;

/* MEM28_CTL register */
typedef struct
{
  __REG32  AHB3_P7RP              : 4;
  __REG32                         : 4;
  __REG32  AHB4_P0RP              : 4;
  __REG32                         : 4;
  __REG32  AHB4_P1RP              : 4;
  __REG32                         : 4;
  __REG32  AHB4_P2RP              : 4;
  __REG32                         : 4;
} __mem28_ctl_bits;

/* MEM29_CTL register */
typedef struct
{
  __REG32  AHB4_P3RP              : 4;
  __REG32                         : 4;
  __REG32  AHB4_P4RP              : 4;
  __REG32                         : 4;
  __REG32  AHB4_P5RP              : 4;
  __REG32                         : 4;
  __REG32  AHB4_P6RP              : 4;
  __REG32                         : 4;
} __mem29_ctl_bits;

/* MEM30_CTL register */
typedef struct
{
  __REG32  AHB4_P7RP              : 4;
  __REG32                         :28;
} __mem30_ctl_bits;

/* MEM34_CTL register */
typedef struct
{
  __REG32                         : 8;
  __REG32  APREBIT                : 4;
  __REG32                         : 4;
  __REG32  CASLAT_LIN             : 4;
  __REG32                         : 4;
  __REG32  CASLAT_LIN_GATE        : 4;
  __REG32                         : 4;
} __mem34_ctl_bits;

/* MEM35_CTL register */
typedef struct
{
  __REG32  COMMAND_AGE_COUNT      : 6;
  __REG32                         : 2;
  __REG32  INITAREF               : 4;
  __REG32                         : 4;
  __REG32  MAX_COL                : 4;
  __REG32                         : 4;
  __REG32  MAX_ROW                : 4;
  __REG32                         : 4;
} __mem35_ctl_bits;

/* MEM36_CTL register */
typedef struct
{
  __REG32  Q_FULLNESS             : 4;
  __REG32                         : 4;
  __REG32  TDAL                   : 4;
  __REG32                         : 4;
  __REG32  TRP                    : 4;
  __REG32                         : 4;
  __REG32  WRR_PARAM_VALUE_ERR    : 4;
  __REG32                         : 4;
} __mem36_ctl_bits;

/* MEM37_CTL register */
typedef struct
{
  __REG32  INT_ACK                : 6;
  __REG32                         : 2;
  __REG32  OCD_ADJUST_PDN_CS0     : 5;
  __REG32                         : 3;
  __REG32  OCD_ADJUST_PUP_CS0     : 5;
  __REG32                         : 3;
  __REG32  TFAW                   : 5;
  __REG32                         : 3;
} __mem37_ctl_bits;

/* MEM38_CTL register */
typedef struct
{
  __REG32  TMRD                   : 5;
  __REG32                         : 3;
  __REG32  TRC                    : 5;
  __REG32                         : 3;
  __REG32  INT_MASK               : 7;
  __REG32                         : 1;
  __REG32  INT_STATUS             : 7;
  __REG32                         : 1;
} __mem38_ctl_bits;

/* MEM39_CTL register */
typedef struct
{
  __REG32  DLL_DQS_DELA0          : 7;
  __REG32                         : 1;
  __REG32  DLL_DQS_DELAY1         : 7;
  __REG32                         :17;
} __mem39_ctl_bits;

/* MEM40_CTL register */
typedef struct
{
  __REG32                         :24;
  __REG32  DQS_OUT_SHIFT          : 7;
  __REG32                         : 1;
} __mem40_ctl_bits;

/* MEM41_CTL register */
typedef struct
{
  __REG32                         :16;
  __REG32  WR_DQS_SHIFT           : 7;
  __REG32                         : 9;
} __mem41_ctl_bits;

/* MEM42_CTL register */
typedef struct
{
  __REG32                         : 8;
  __REG32  TRAS_MIN               : 8;
  __REG32  TRCD_INT               : 8;
  __REG32  TRFC                   : 8;
} __mem42_ctl_bits;

/* MEM43_CTL register */
typedef struct
{
  __REG32  AHB0_PRIORITY_RELAX    :10;
  __REG32                         : 6;
  __REG32  AHB1_PRIORITY_RELAX    :10;
  __REG32  TRFC                   : 6;
} __mem43_ctl_bits;

/* MEM44_CTL register */
typedef struct
{
  __REG32  AHB2_PRIORITY_RELAX    :10;
  __REG32                         : 6;
  __REG32  AHB3_PRIORITY_RELAX    :10;
  __REG32  TRFC                   : 6;
} __mem44_ctl_bits;

/* MEM45_CTL register */
typedef struct
{
  __REG32  AHB4_PRIORITY_RELAX    :10;
  __REG32                         :22;
} __mem45_ctl_bits;

/* MEM46_CTL register */
typedef struct
{
  __REG32                         :16;
  __REG32  OUT_OF_RANGE_LENGTH    :10;
  __REG32                         : 6;
} __mem46_ctl_bits;

/* MEM47_CTL register */
typedef struct
{
  __REG32  AHB0_RDCNT             :11;
  __REG32                         : 5;
  __REG32  AHB0_WRCNT             :10;
  __REG32                         : 6;
} __mem47_ctl_bits;

/* MEM48_CTL register */
typedef struct
{
  __REG32  AHB1_RDCNT             :11;
  __REG32                         : 5;
  __REG32  AHB1_WRCNT             :10;
  __REG32                         : 6;
} __mem48_ctl_bits;

/* MEM49_CTL register */
typedef struct
{
  __REG32  AHB2_RDCNT             :11;
  __REG32                         : 5;
  __REG32  AHB2_WRCNT             :10;
  __REG32                         : 6;
} __mem49_ctl_bits;

/* MEM50_CTL register */
typedef struct
{
  __REG32  AHB3_RDCNT             :11;
  __REG32                         : 5;
  __REG32  AHB3_WRCNT             :10;
  __REG32                         : 6;
} __mem50_ctl_bits;

/* MEM51_CTL register */
typedef struct
{
  __REG32  AHB4_RDCNT             :11;
  __REG32                         : 5;
  __REG32  AHB4_WRCNT             :10;
  __REG32                         : 6;
} __mem51_ctl_bits;

/* MEM54_CTL register */
typedef struct
{
  __REG32  TREF                   :14;
  __REG32                         :18;
} __mem54_ctl_bits;

/* MEM55_CTL register */
typedef struct
{
  __REG32  EMRS3_DATA             :15;
  __REG32                         :17;
} __mem55_ctl_bits;

/* MEM56_CTL register */
typedef struct
{
  __REG32  TDLL                   :16;
  __REG32  TRAS_MAX               :16;
} __mem56_ctl_bits;

/* MEM57_CTL register */
typedef struct
{
  __REG32  TXSNR                  :16;
  __REG32  TXSR                   :16;
} __mem57_ctl_bits;

/* MEM58_CTL register */
typedef struct
{
  __REG32  VERSION                :16;
  __REG32                         :16;
} __mem58_ctl_bits;

/* MEM59_CTL register */
typedef struct
{
  __REG32  TINIT                  :24;
  __REG32                         : 8;
} __mem59_ctl_bits;

/* MEM61_CTL register */
typedef struct
{
  __REG32  OUT_OF_RANGE_ADDR      : 2;
  __REG32                         :30;
} __mem61_ctl_bits;

/* MEM65_CTL register */
typedef struct
{
  __REG32                         :16;
  __REG32  DLL_DQS_DELAY_BYPASS_0 :10;
  __REG32                         : 6;
} __mem65_ctl_bits;

/* MEM66_CTL register */
typedef struct
{
  __REG32  DLL_DQS_DELAY_BYPASS_1 :10;
  __REG32                         : 6;
  __REG32  DLL_INCREMENT          :10;
  __REG32                         : 6;
} __mem66_ctl_bits;

/* MEM67_CTL register */
typedef struct
{
  __REG32  DLL_LOCK               :10;
  __REG32                         : 6;
  __REG32  DLL_START_POINT        :10;
  __REG32                         : 6;
} __mem67_ctl_bits;

/* MEM68_CTL register */
typedef struct
{
  __REG32  DQS_OUT_SHIFT_BYPASS   :10;
  __REG32                         : 6;
  __REG32  WR_DQS_SHIFT_BYPASS    :10;
  __REG32                         : 6;
} __mem68_ctl_bits;

/* MEM100_CTL register */
typedef struct
{
  __REG32  ACTIVE_AGING           : 1;
  __REG32                         : 7;
  __REG32  BIG_ENDIAN_ENABLE      : 1;
  __REG32                         : 7;
  __REG32  DRIVE_DQ_DQS           : 1;
  __REG32                         : 7;
  __REG32  ENABLE_QUICK_SREFRESH  : 1;
  __REG32                         : 7;
} __mem100_ctl_bits;

/* MEM101_CTL register */
typedef struct
{
  __REG32  EN_LOWPOWER_MODE       : 1;
  __REG32                         : 7;
  __REG32  PWRUP_SREFRESH_EXIT    : 1;
  __REG32                         : 7;
  __REG32  RD2RD_TURN             : 1;
  __REG32                         : 7;
  __REG32  SWAP_ENABLE            : 1;
  __REG32                         : 7;
} __mem101_ctl_bits;

/* MEM102_CTL register */
typedef struct
{
  __REG32  TREF_ENABLE            : 1;
  __REG32                         : 7;
  __REG32  LOWPOWER_REFRESH_ENABLE: 2;
  __REG32                         : 6;
  __REG32  CKE_DELAY              : 3;
  __REG32                         : 5;
  __REG32  LOWPOWER_AUTO_ENABLE   : 5;
  __REG32                         : 3;
} __mem102_ctl_bits;

/* MEM103_CTL register */
typedef struct
{
  __REG32  LOWPOWER_CONTROL       : 5;
  __REG32                         : 3;
  __REG32  EMRS1_DATA             :15;
  __REG32                         : 9;
} __mem103_ctl_bits;

/* MEM104_CTL register */
typedef struct
{
  __REG32  EMRS2_DATA0            :15;
  __REG32                         : 1;
  __REG32  EMRS2_DATA1            :15;
  __REG32                         : 1;
} __mem104_ctl_bits;

/* MEM105_CTL register */
typedef struct
{
  __REG32  LOWPOWER_EXT_CNT       :16;
  __REG32  LOWPOWER_INT_CNT       :16;
} __mem105_ctl_bits;

/* MEM106_CTL register */
typedef struct
{
  __REG32  LOWPOWER_PWDWN_CNT     :16;
  __REG32  LOWPOWER_RFSH_HOLD     :16;
} __mem106_ctl_bits;

/* MEM107_CTL register */
typedef struct
{
  __REG32  LOWPOWER_SRFSH_CNT     :16;
  __REG32  TCPD                   :16;
} __mem107_ctl_bits;

/* MEM108_CTL register */
typedef struct
{
  __REG32  TPDEX                  :16;
  __REG32                         :16;
} __mem108_ctl_bits;

/* SoC_CFG_CTR register */
typedef struct
{
  __REG32  SoC_cfg                : 6;
  __REG32  Fixed_Value            :13;
  __REG32  boot_sel               : 1;
  __REG32                         :12;
} __soc_cfg_ctr_bits;

/* DIAG_CFG_CTR register */
typedef struct
{
  __REG32                         : 4;
  __REG32  SOC_dbg6               : 2;
  __REG32                         : 5;
  __REG32  sys_error              : 1;
  __REG32                         : 3;
  __REG32  debug_freez            : 1;
  __REG32                         :16;
} __diag_cfg_ctr_bits;

/* PLL 1/2_CTR register */
typedef struct
{
  __REG32  pll_lock               : 1;
  __REG32  pll_resetn             : 1;
  __REG32  pll_enable             : 1;
  __REG32  pll_control1           : 6;
  __REG32                         :23;
} __pll_ctr_bits;

/* PLL1/2_FRQ registers */
typedef struct
{
  __REG32  pll_prediv_N           : 8;
  __REG32  pll_postdiv_P          : 3;
  __REG32                         : 5;
  __REG32  pll_fbkdiv_M           :16;
} __pll_frq_bits;

/* PLL1/2_MOD registers */
typedef struct
{
  __REG32  pll_slope              :16;
  __REG32  pll_modperiod          :13;
  __REG32                         : 3;
} __pll_mod_bits;

/* PLL_CLK_CFG register */
typedef struct
{
  __REG32  pll1_enb_clkout        : 1;
  __REG32  pll2_enb_clkout        : 1;
  __REG32  pll3_enb_clkout        : 1;
  __REG32                         :13;
  __REG32  sys_pll1_lock          : 1;
  __REG32  sys_pll2_lock          : 1;
  __REG32  usb_pll_lock           : 1;
  __REG32  mem_dll_lock           : 1;
  __REG32  pll1_clk_sel           : 3;
  __REG32                         : 1;
  __REG32  pll2_clk_sel           : 3;
  __REG32                         : 1;
  __REG32  mctr_clk_sel           : 3;
  __REG32                         : 1;
} __pll_clk_cfg_bits;

/* CORE_CLK_CFG register */
typedef struct
{
  __REG32  pclk_ratio_arm1        : 2;
  __REG32                         : 2;
  __REG32  pclk_ratio_basc        : 2;
  __REG32                         : 2;
  __REG32  pclk_ratio_lwsp        : 2;
  __REG32  hclk_divsel            : 2;
  __REG32                         : 6;
  __REG32  ras_synt34_clksel      : 1;
  __REG32  Osci24_div_en          : 1;
  __REG32  Osci24_div_ratio       : 2;
  __REG32                         :10;
} __core_clk_cfg_bits;

/* PRPH_CLK_CFG register */
typedef struct
{
  __REG32  xtaltimeen             : 1;
  __REG32  plltimeen              : 1;
  __REG32                         : 2;
  __REG32  uart_clksel            : 1;
  __REG32  irda_clksel            : 2;
  __REG32  rtc_disable            : 1;
  __REG32  gptmr1_clksel          : 1;
  __REG32                         : 2;
  __REG32  gptmr2_clksel          : 1;
  __REG32  gptrmr3_clksel         : 1;
  __REG32  gptmr1_freez           : 1;
  __REG32                         : 2;
  __REG32  gptmr2_freez           : 1;
  __REG32  gptmr3_freez           : 1;
  __REG32                         :14;
} __prph_clk_cfg_bits;

/* PERIP1_CLK_ENB register */
typedef struct
{
  __REG32  arm_enb                : 1;
  __REG32  arm_clkenb             : 1;
  __REG32                         : 1;
  __REG32  uart_clkenb            : 1;
  __REG32                         : 1;
  __REG32  ssp_clkenb             : 1;
  __REG32                         : 1;
  __REG32  i2c_clkenb             : 1;
  __REG32  jpeg_clkenb            : 1;
  __REG32                         : 1;
  __REG32  firda_clkenb           : 1;
  __REG32  GPT2_clkenb            : 1;
  __REG32  GPT3_clkenb            : 1;
  __REG32                         : 2;
  __REG32  adc_clkenb             : 1;
  __REG32                         : 1;
  __REG32  rtc_clkenb             : 1;
  __REG32  GPIO_clkenb            : 1;
  __REG32  DMA_clkenb             : 1;
  __REG32  rom_clkenb             : 1;
  __REG32  smi_clkenb             : 1;
  __REG32                         : 1;
  __REG32  MAC_clkenb             : 1;
  __REG32  usbdev_clkenb          : 1;
  __REG32  usbh1_clkenb           : 1;
  __REG32                         : 1;
  __REG32  ddr_clkenb             : 1;
  __REG32                         : 1;
  __REG32  ddr_core_enb           : 1;
  __REG32                         : 1;
  __REG32  C3_clock_enb           : 1;
} __perip1_clk_enb_bits;

/* RAS_CLK_ENB register */
typedef struct
{
  __REG32  hclk_clkenb            : 1;
  __REG32  pll1_clkenb            : 1;
  __REG32  pclkappl_clkenb        : 1;
  __REG32  clk32K_clkenb          : 1;
  __REG32  Clk24M_clkenb          : 1;
  __REG32  clk48M_clkenb          : 1;
  __REG32                         : 1;
  __REG32  pll2_clkenb            : 1;
  __REG32  ras_synt1_clkenb       : 1;
  __REG32  ras_synt2_clkenb       : 1;
  __REG32  ras_synt3_clkenb       : 1;
  __REG32  ras_synt4_clkenb       : 1;
  __REG32  pl_gpck1_clkenb        : 1;
  __REG32  pl_gpck2_clkenb        : 1;
  __REG32  pl_gpck3_clkenb        : 1;
  __REG32  pl_gpck4_clkenb        : 1;
  __REG32                         :16;
} __ras_clk_enb_bits;

/* AMEM_CFG_CTRL register */
typedef struct
{
  __REG32  amem_clk_enb           : 1;
  __REG32  amem_clk_sel           : 3;
  __REG32  amem_synt_enb          : 1;
  __REG32                         :10;
  __REG32  amem_rst               : 1;
  __REG32  amem_ydiv              : 8;
  __REG32  amem_xdiv              : 8;
} __amem_cfg_ctrl_bits;

/* Auxiliary clock synthesizer register */
typedef struct
{
  __REG32  synt_ydiv              :12;
  __REG32                         : 4;
  __REG32  synt_xdiv              :12;
  __REG32                         : 2;
  __REG32  synt_clkout_sel        : 1;
  __REG32  synt_clk_enb           : 1;
} __irda_clk_synt_bits;

/* PERIP1_SOF_RST register */
typedef struct
{
  __REG32  arm1_enbr              : 1;
  __REG32  arm1_swrst             : 1;
  __REG32                         : 1;
  __REG32  uart_swrst             : 1;
  __REG32                         : 1;
  __REG32  ssp_swrst              : 1;
  __REG32                         : 1;
  __REG32  i2c_swrst              : 1;
  __REG32  jpeg_swrst             : 1;
  __REG32                         : 1;
  __REG32  firda_swrst            : 1;
  __REG32  gptm2_swrst            : 1;
  __REG32  gptm3_swrst            : 1;
  __REG32                         : 2;
  __REG32  adc_swrst              : 1;
  __REG32                         : 1;
  __REG32  rtc_swrst              : 1;
  __REG32  gpio_swrst             : 1;
  __REG32  DMA_swrst              : 1;
  __REG32  rom_swrst              : 1;
  __REG32  smi_swrst              : 1;
  __REG32                         : 1;
  __REG32  MAC_swrst              : 1;
  __REG32  usbdev_swrst           : 1;
  __REG32  usbh1_ohci_swrst       : 1;
  __REG32  usbh1_ehci_swrst       : 1;
  __REG32  ddr_swrst              : 1;
  __REG32  ram_swrst              : 1;
  __REG32  ddr_core_enbr          : 1;
  __REG32                         : 1;
  __REG32  C3_reset               : 1;
} __perip1_sof_rst_bits;

/* RAS_SOF_RST register */
typedef struct
{
  __REG32  hclk_swrst             : 1;
  __REG32  pll1_swrst             : 1;
  __REG32  pclkappl_swrst         : 1;
  __REG32  Clk32K_swrst           : 1;
  __REG32  Clk24M_swrst           : 1;
  __REG32  clk48M_swrst           : 1;
  __REG32  clk125M_swrst          : 1;
  __REG32  pll2_swrst             : 1;
  __REG32  ras_synt1_swrst        : 1;
  __REG32  ras_synt2_swrst        : 1;
  __REG32  ras_synt3_swrst        : 1;
  __REG32  ras_synt4_swrst        : 1;
  __REG32  pl_gpck1_swrst         : 1;
  __REG32  pl_gpck2_swrst         : 1;
  __REG32  pl_gpck3_swrst         : 1;
  __REG32  pl_gpck4_swrst         : 1;
  __REG32                         :16;
} __ras_sof_rst_bits;

/* PRSC1/2/3_CLK_CFG register */
typedef struct
{
  __REG32  presc_m                :12;
  __REG32  presc_n                : 4;
  __REG32  RFU                    :16;
} __prsc_clk_cfg_bits;

/* ICM 1-9_ARB_CFG register bit assignments */
typedef struct
{
  __REG32  mtx_fix_pry_lyr0       : 3;
  __REG32  mtx_fix_pry_lyr1       : 3;
  __REG32  mtx_fix_pry_lyr2       : 3;
  __REG32  mtx_fix_pry_lyr3       : 3;
  __REG32  mtx_fix_pry_lyr4       : 3;
  __REG32  mtx_fix_pry_lyr5       : 3;
  __REG32  mtx_fix_pry_lyr6       : 3;
  __REG32  mtx_fix_pry_lyr7       : 3;
  __REG32                         : 4;
  __REG32  mxt_rndrb_pry_lyr      : 3;
  __REG32  mtx_arb_type           : 1;
} __icm_arb_cfg_bits;

/* DMA_CHN_CFG register */
typedef struct
{
  __REG32  dma_cfg_chan00         : 2;
  __REG32  dma_cfg_chan01         : 2;
  __REG32  dma_cfg_chan02         : 2;
  __REG32  dma_cfg_chan03         : 2;
  __REG32  dma_cfg_chan04         : 2;
  __REG32  dma_cfg_chan05         : 2;
  __REG32  dma_cfg_chan06         : 2;
  __REG32  dma_cfg_chan07         : 2;
  __REG32  dma_cfg_chan08         : 2;
  __REG32  dma_cfg_chan09         : 2;
  __REG32  dma_cfg_chan10         : 2;
  __REG32  dma_cfg_chan11         : 2;
  __REG32  dma_cfg_chan12         : 2;
  __REG32  dma_cfg_chan13         : 2;
  __REG32  dma_cfg_chan14         : 2;
  __REG32  dma_cfg_chan15         : 2;
} __dma_chn_cfg_bits;

/* USB2_PHY_CFG register */
typedef struct
{
  __REG32  PLL_pwdn               : 1;
  __REG32                         : 2;
  __REG32  usbh_overcur           : 1;
  __REG32                         :28;
} __usb2_phy_cfg_bits;

/* MAC_CFG_CTR register */
typedef struct
{
  __REG32  mili_reverse           : 1;
  __REG32                         : 1;
  __REG32  MAC_clk_sel            : 2;
  __REG32  MAC_synt_enb           : 1;
  __REG32                         :27;
} __mac_cfg_ctr_bits;

/* PRC1-2_LOCK_CTR register */
typedef struct
{
  __REG32  lock_request           : 4;
  __REG32  lock_reset             : 4;
  __REG32                         : 9;
  __REG32  sts_loc_lock_1         : 1;
  __REG32  sts_loc_lock_2         : 1;
  __REG32  sts_loc_lock_3         : 1;
  __REG32  sts_loc_lock_4         : 1;
  __REG32  sts_loc_lock_5         : 1;
  __REG32  sts_loc_lock_6         : 1;
  __REG32  sts_loc_lock_7         : 1;
  __REG32  sts_loc_lock_8         : 1;
  __REG32  sts_loc_lock_9         : 1;
  __REG32  sts_loc_lock_10        : 1;
  __REG32  sts_loc_lock_11        : 1;
  __REG32  sts_loc_lock_12        : 1;
  __REG32  sts_loc_lock_13        : 1;
  __REG32  sts_loc_lock_14        : 1;
  __REG32  sts_loc_lock_15        : 1;
} __prc_lock_ctr_bits;

/* PRC1-2_IRQ_CTR register */
typedef struct
{
  __REG32                         :16;
  __REG32  int2_req_prc1_1        : 1;
  __REG32                         :15;
} __prc_irq_ctr_bits;

/* Powerdown_CFG_CTR register */
typedef struct
{
  __REG32  wakeup_fiq_enb         : 1;
  __REG32                         :31;
} __powerdown_cfg_ctr_bits;

/* COMPSSTL_1V8_CFG/DDR_2V5_COMPENSATION register */
typedef struct
{
  __REG32  compen                 : 1;
  __REG32  comptq                 : 1;
  __REG32  freeze                 : 1;
  __REG32  accurate               : 1;
  __REG32  COMPOK                 : 1;
  __REG32                         :11;
  __REG32  nasrc                  : 7;
  __REG32                         : 1;
  __REG32  rasrc                  : 7;
  __REG32  TQ                     : 1;
} __compsstl_1v8_cfg_bits;

/* DDR_PAD register */
typedef struct
{
  __REG32  DDR_LOW_POWER_DDR2_mode  : 1;
  __REG32  PROG_b                   : 1;
  __REG32  PROG_a                   : 1;
  __REG32  S_W_mode                 : 1;
  __REG32  PU_sel                   : 1;
  __REG32  PDN_sel                  : 1;
  __REG32  CLK_PU_sel               : 1;
  __REG32  CLK_PDN_sel              : 1;
  __REG32  DQS_PU_sel               : 1;
  __REG32  DQS_PDN_sel              : 1;
  __REG32  ENZI                     : 1;
  __REG32                           : 1;
  __REG32  GATE_OPEN_mode           : 1;
  __REG32  REFSSTL                  : 1;
  __REG32  DDR_EN_PAD               : 1;
  __REG32  DDR_SW_mode              : 4;
  __REG32                           :13;
} __ddr_pad_bits;

/* BIST1_CFG_CTR register */
typedef struct
{
  __REG32  rbact0                   : 1;
  __REG32  rbact1                   : 1;
  __REG32  rbact2                   : 1;
  __REG32  rbact3                   : 1;
  __REG32  rbact4                   : 1;
  __REG32  rbact5                   : 1;
  __REG32  rbact6                   : 1;
  __REG32  rbact7                   : 1;
  __REG32  rbact8                   : 1;
  __REG32  rbact9                   : 1;
  __REG32  rbact10                  : 1;
  __REG32  rbact11                  : 1;
  __REG32  rbact12                  : 1;
  __REG32  rbact13                  : 1;
  __REG32  rbact14                  : 1;
  __REG32                           : 9;
  __REG32  bist_iddq                : 1;
  __REG32  bist_ret                 : 1;
  __REG32  bist_debug               : 1;
  __REG32  bist_tm                  : 1;
  __REG32  bist_rst                 : 1;
  __REG32                           : 2;
  __REG32  bist_res_rst             : 1;
} __bist1_cfg_ctr_bits;

/* BIST2_CFG_CTR register */
typedef struct
{
  __REG32  rbact0                   : 1;
  __REG32  rbact1                   : 1;
  __REG32  rbact2                   : 1;
  __REG32  rbact3                   : 1;
  __REG32                           :20;
  __REG32  bist_iddq                : 1;
  __REG32  bist_ret                 : 1;
  __REG32  bist_debug               : 1;
  __REG32  bist_tm                  : 1;
  __REG32  bist_rst                 : 1;
  __REG32                           : 2;
  __REG32  bist_res_rst             : 1;
} __bist2_cfg_ctr_bits;

/* BIST3_CFG_CTR register */
typedef struct
{
  __REG32  rbact0                   : 1;
  __REG32  rbact1                   : 1;
  __REG32  rbact2                   : 1;
  __REG32                           :21;
  __REG32  bist_iddq                : 1;
  __REG32  bist_ret                 : 1;
  __REG32  bist_debug               : 1;
  __REG32  bist_tm                  : 1;
  __REG32  bist_rst                 : 1;
  __REG32                           : 2;
  __REG32  bist_res_rst             : 1;
} __bist3_cfg_ctr_bits;

/* BIST4_CFG_CTR register */
typedef struct
{
  __REG32  rbact0                   : 1;
  __REG32  rbact1                   : 1;
  __REG32  rbact2                   : 1;
  __REG32  rbact3                   : 1;
  __REG32  rbact4                   : 1;
  __REG32  rbact5                   : 1;
  __REG32  rbact6                   : 1;
  __REG32  rbact7                   : 1;
  __REG32                           :16;
  __REG32  bist_iddq                : 1;
  __REG32  bist_ret                 : 1;
  __REG32  bist_debug               : 1;
  __REG32  bist_tm                  : 1;
  __REG32  bist_rst                 : 1;
  __REG32                           : 2;
  __REG32  bist_res_rst             : 1;
} __bist4_cfg_ctr_bits;

/* BIST1_STS_RES register */
typedef struct
{
  __REG32  bbad0                    : 1;
  __REG32  bbad1                    : 1;
  __REG32  bbad2                    : 1;
  __REG32  bbad3                    : 1;
  __REG32  bbad4                    : 1;
  __REG32  bbad5                    : 1;
  __REG32  bbad6                    : 1;
  __REG32  bbad7                    : 1;
  __REG32  bbad8                    : 1;
  __REG32  bbad9                    : 1;
  __REG32  bbad10                   : 1;
  __REG32  bbad11                   : 1;
  __REG32  bbad12                   : 1;
  __REG32  bbad13                   : 1;
  __REG32  bbad14                   : 1;
  __REG32                           :16;
  __REG32  bist_end                 : 1;
} __bist1_sts_res_bits;

/* BIST2_STS_RES register */
typedef struct
{
  __REG32  bbad0                    : 1;
  __REG32  bbad1                    : 1;
  __REG32  bbad2                    : 1;
  __REG32  bbad3                    : 1;
  __REG32  bbad4                    : 1;
  __REG32  bbad5                    : 1;
  __REG32  bbad6                    : 1;
  __REG32  bbad7                    : 1;
  __REG32  bbad8                    : 1;
  __REG32  bbad9                    : 1;
  __REG32  bbad10                   : 1;
  __REG32  bbad11                   : 1;
  __REG32  bbad12                   : 1;
  __REG32  bbad13                   : 1;
  __REG32  bbad14                   : 1;
  __REG32                           :16;
  __REG32  bist_end                 : 1;
} __bist2_sts_res_bits;

/* BIST3_STS_RES register */
typedef struct
{
  __REG32  bbad0                    : 1;
  __REG32  bbad1                    : 1;
  __REG32  bbad2                    : 1;
  __REG32  bbad3                    : 1;
  __REG32  bbad4                    : 1;
  __REG32  bbad5                    : 1;
  __REG32  bbad6                    : 1;
  __REG32  bbad7                    : 1;
  __REG32  bbad8                    : 1;
  __REG32  bbad9                    : 1;
  __REG32  bbad10                   : 1;
  __REG32  bbad11                   : 1;
  __REG32  bbad12                   : 1;
  __REG32  bbad13                   : 1;
  __REG32                           :17;
  __REG32  bist_end                 : 1;
} __bist3_sts_res_bits;

/* BIST4_STS_RES register */
typedef struct
{
  __REG32  bbad0                    : 1;
  __REG32  bbad1                    : 1;
  __REG32  bbad2                    : 1;
  __REG32  bbad3                    : 1;
  __REG32  bbad4                    : 1;
  __REG32  bbad5                    : 1;
  __REG32  bbad6                    : 1;
  __REG32  bbad7                    : 1;
  __REG32  bbad8                    : 1;
  __REG32  bbad9                    : 1;
  __REG32  bbad10                   : 1;
  __REG32  bbad11                   : 1;
  __REG32  bbad12                   : 1;
  __REG32  bbad13                   : 1;
  __REG32                           :17;
  __REG32  bist_end                 : 1;
} __bist4_sts_res_bits;

/* SYSERR_CFG_CTR register */
typedef struct
{
  __REG32  int_error_enb            : 1;
  __REG32  int_error_rst            : 1;
  __REG32  int_error                : 1;
  __REG32                           : 1;
  __REG32  pll_err_enb              : 1;
  __REG32                           : 1;
  __REG32  wdg_err_enb              : 1;
  __REG32                           : 1;
  __REG32  usb_err_enb              : 1;
  __REG32  mem_err_enb              : 1;
  __REG32  DMA_err_enb              : 1;
  __REG32                           : 1;
  __REG32  sys_pll1_err             : 1;
  __REG32  sys_pll2_err             : 1;
  __REG32  usb_pll_err              : 1;
  __REG32  mem_dll_err              : 1;
  __REG32                           : 6;
  __REG32  arm1_wdg_err             : 1;
  __REG32  arm2_wdg_err             : 1;
  __REG32  usbdv_err                : 1;
  __REG32  usbh1_err                : 1;
  __REG32  usbh2_err                : 1;
  __REG32  Mem_err                  : 1;
  __REG32  DMA_err                  : 1;
  __REG32                           : 3;
} __syserr_cfg_ctr_bits;

/* USB_TUN_PRM register */
typedef struct
{
  __REG32  TXRISETUNE               : 1;
  __REG32  TXHSXVTUNE               : 2;
  __REG32  TXPREEMPHASISTUNE        : 1;
  __REG32  TXVREFTUNE               : 4;
  __REG32  TXFSLSTUNE               : 4;
  __REG32  SQRXTUNE                 : 3;
  __REG32  COMPDISTUNE              : 3;
  __REG32                           :14;
} __usb_tun_prm_bits;

/* PLGPIO0_PAD_PRG register */
typedef struct
{
  __REG32  SLEW_0                   : 1;
  __REG32  DRV_0                    : 2;
  __REG32  PUP_0                    : 1;
  __REG32  PDN_0                    : 1;
  __REG32  SLEW_1                   : 1;
  __REG32  DRV_1                    : 2;
  __REG32  PUP_1                    : 1;
  __REG32  PDN_1                    : 1;
  __REG32  SLEW_2                   : 1;
  __REG32  DRV_2                    : 2;
  __REG32  PUP_2                    : 1;
  __REG32  PDN_2                    : 1;
  __REG32  SLEW_3                   : 1;
  __REG32  DRV_3                    : 2;
  __REG32  PUP_3                    : 1;
  __REG32  PDN_3                    : 1;
  __REG32  SLEW_4                   : 1;
  __REG32  DRV_4                    : 2;
  __REG32  PUP_4                    : 1;
  __REG32  PDN_4                    : 1;
  __REG32  SLEW_5                   : 1;
  __REG32  DRV_5                    : 2;
  __REG32  PUP_5                    : 1;
  __REG32  PDN_5                    : 1;
  __REG32  PUP_UART                 : 1;
  __REG32  PDN_UART                 : 1;
} __plgpio0_pad_prg_bits;

/* PLGPIO1_PAD_PRG register */
typedef struct
{
  __REG32  SLEW_6                   : 1;
  __REG32  DRV_6                    : 2;
  __REG32  PUP_6                    : 1;
  __REG32  PDN_6                    : 1;
  __REG32  SLEW_7                   : 1;
  __REG32  DRV_7                    : 2;
  __REG32  PUP_7                    : 1;
  __REG32  PDN_7                    : 1;
  __REG32  SLEW_8                   : 1;
  __REG32  DRV_8                    : 2;
  __REG32  PUP_8                    : 1;
  __REG32  PDN_8                    : 1;
  __REG32  SLEW_9                   : 1;
  __REG32  DRV_9                    : 2;
  __REG32  PUP_9                    : 1;
  __REG32  PDN_9                    : 1;
  __REG32  SLEW_10                  : 1;
  __REG32  DRV_10                   : 2;
  __REG32  PUP_10                   : 1;
  __REG32  PDN_10                   : 1;
  __REG32  SLEW_11                  : 1;
  __REG32  DRV_11                   : 2;
  __REG32  PUP_11                   : 1;
  __REG32  PDN_11                   : 1;
  __REG32  PUP_I2C                  : 1;
  __REG32  PDN_I2C                  : 1;
} __plgpio1_pad_prg_bits;

/* PLGPIO2_PAD_PRG register */
typedef struct
{
  __REG32  SLEW_12                  : 1;
  __REG32  DRV_12                   : 2;
  __REG32  PUP_12                   : 1;
  __REG32  PDN_12                   : 1;
  __REG32  SLEW_13                  : 1;
  __REG32  DRV_13                   : 2;
  __REG32  PUP_13                   : 1;
  __REG32  PDN_13                   : 1;
  __REG32  SLEW_14                  : 1;
  __REG32  DRV_14                   : 2;
  __REG32  PUP_14                   : 1;
  __REG32  PDN_14                   : 1;
  __REG32  SLEW_15                  : 1;
  __REG32  DRV_15                   : 2;
  __REG32  PUP_15                   : 1;
  __REG32  PDN_15                   : 1;
  __REG32  SLEW_16                  : 1;
  __REG32  DRV_16                   : 2;
  __REG32  PUP_16                   : 1;
  __REG32  PDN_16                   : 1;
  __REG32  SLEW_17                  : 1;
  __REG32  DRV_17                   : 2;
  __REG32  PUP_17                   : 1;
  __REG32  PDN_17                   : 1;
  __REG32  PUP_ETHERNET             : 1;
  __REG32  PDN_ETHERNET             : 1;
} __plgpio2_pad_prg_bits;

/* PLGPIO3_PAD_PRG register */
typedef struct
{
  __REG32  SLEW_18                  : 1;
  __REG32  DRV_18                   : 2;
  __REG32  PUP_18                   : 1;
  __REG32  PDN_18                   : 1;
  __REG32  SLEW_19                  : 1;
  __REG32  DRV_19                   : 2;
  __REG32  PUP_19                   : 1;
  __REG32  PDN_19                   : 1;
  __REG32  SLEW_20                  : 1;
  __REG32  DRV_20                   : 2;
  __REG32  PUP_20                   : 1;
  __REG32  PDN_20                   : 1;
  __REG32  SLEW_21                  : 1;
  __REG32  DRV_21                   : 2;
  __REG32  PUP_21                   : 1;
  __REG32  PDN_21                   : 1;
  __REG32  SLEW_22                  : 1;
  __REG32  DRV_22                   : 2;
  __REG32  PUP_22                   : 1;
  __REG32  PDN_22                   : 1;
  __REG32  SLEW_23                  : 1;
  __REG32  DRV_23                   : 2;
  __REG32  PUP_23                   : 1;
  __REG32  PDN_23                   : 1;
  __REG32                           : 2;
} __plgpio3_pad_prg_bits;

/* PLGPIO4_PAD_PRG register */
typedef struct
{
  __REG32                           : 1;
  __REG32  DRV_24                   : 2;
  __REG32  PUP_24                   : 1;
  __REG32  PDN_24                   : 1;
  __REG32  SLEW_CLK1                : 1;
  __REG32  DRV_CLK1                 : 2;
  __REG32  PUP_CLK1                 : 1;
  __REG32  PDN_CLK1                 : 1;
  __REG32  SLEW_CLK2                : 1;
  __REG32  DRV_CLK2                 : 2;
  __REG32  PUP_CLK2                 : 1;
  __REG32  PDN_CLK2                 : 1;
  __REG32  SLEW_CLK3                : 1;
  __REG32  DRV_CLK3                 : 2;
  __REG32  PUP_CLK3                 : 1;
  __REG32  PDN_CLK3                 : 1;
  __REG32  SLEW_CLK4                : 1;
  __REG32  DRV_CLK4                 : 2;
  __REG32  PUP_CLK4                 : 1;
  __REG32  PDN_CLK4                 : 1;
  __REG32                           : 7;
} __plgpio4_pad_prg_bits;

/* SSPCR0 register */
typedef struct
{
  __REG16  DSS                  : 4;
  __REG16  FRF                  : 2;
  __REG16  SPO                  : 1;
  __REG16  SPH                  : 1;
  __REG16  SCR                  : 8;
} __sspcr0_bits;

/* SSPCR1 register */
typedef struct
{
  __REG16  LBM                  : 1;
  __REG16  SSE                  : 1;
  __REG16  MS                   : 1;
  __REG16  SOD                  : 1;
  __REG16                       :12;
} __sspcr1_bits;

/* SSPSR register */
typedef struct
{
  __REG16  TFE                  : 1;
  __REG16  TNF                  : 1;
  __REG16  RNE                  : 1;
  __REG16  RFF                  : 1;
  __REG16  BSY                  : 1;
  __REG16                       :11;
} __sspsr_bits;

/* SSPCPSR register */
typedef struct
{
  __REG16  CPSDVSR              : 8;
  __REG16                       : 8;
} __sspcpsr_bits;

/* SSPIMSC register */
typedef struct
{
  __REG16  RORIM                : 1;
  __REG16  RTIM                 : 1;
  __REG16  RXIM                 : 1;
  __REG16  TXIM                 : 1;
  __REG16                       :12;
} __sspimsc_bits;

/* SSPRIS register */
typedef struct
{
  __REG16  RORRIS               : 1;
  __REG16  RTRIS                : 1;
  __REG16  RXRIS                : 1;
  __REG16  TXRIS                : 1;
  __REG16                       :12;
} __sspris_bits;

/* SSPMIS Register */
typedef struct
{
  __REG16  RORMIS               : 1;
  __REG16  RTMIS                : 1;
  __REG16  RXMIS                : 1;
  __REG16  TXMIS                : 1;
  __REG16                       :12;
} __sspmis_bits;

/* SSPICR register */
typedef struct
{
  __REG16  RORIC                : 1;
  __REG16  RTIC                 : 1;
  __REG16                       :14;
} __sspicr_bits;

/* SSPDMACR register */
typedef struct
{
  __REG16  RXDMAE               : 1;
  __REG16  TXDMAE               : 1;
  __REG16                       :14;
} __sspdmacr_bits;

/* PHERIPHID0 register */
typedef struct
{
  __REG16  PartNumber0          : 8;
  __REG16                       : 8;
} __sspperiphid0_bits;

/* PHERIPHID1 register */
typedef struct
{
  __REG16  PartNumber1          : 4;
  __REG16  Designer0            : 4;
  __REG16                       : 8;
} __sspperiphid1_bits;

/* PHERIPHID2 register */
typedef struct
{
  __REG16  Designer1            : 4;
  __REG16  Revision             : 4;
  __REG16                       : 8;
} __sspperiphid2_bits;

/* PHERIPHID3 register */
typedef struct
{
  __REG16  Configuration        : 8;
  __REG16                       : 8;
} __sspperiphid3_bits;

/* PCELLID0 register */
typedef struct
{
  __REG16  PCELLID0             : 8;
  __REG16                       : 8;
} __sspcellid0_bits;

/* PCELLID1 register */
typedef struct
{
  __REG16  PCELLID1             : 8;
  __REG16                       : 8;
} __sspcellid1_bits;

/* PCELLID2 register */
typedef struct
{
  __REG16  PCELLID2             : 8;
  __REG16                       : 8;
} __sspcellid2_bits;

/* PCELLID3 register */
typedef struct
{
  __REG16  PCELLID3             : 8;
  __REG16                       : 8;
} __sspcellid3_bits;

/* SCCTRL register */
typedef struct
{
  __REG32  ModeCtrl             : 3;
  __REG32  ModeStatus           : 4;
  __REG32                       : 1;
  __REG32  RemapClear           : 1;
  __REG32  RemapStat            : 1;
  __REG32                       : 2;
  __REG32  HCLKDivSel           : 3;
  __REG32  TimerEn0Sel          : 1;
  __REG32  TimerEn0Ov           : 1;
  __REG32  TimerEn1Sel          : 1;
  __REG32  TimerEn1Ov           : 1;
  __REG32  TimerEn2Sel          : 1;
  __REG32  TimerEn2Ov           : 1;
  __REG32                       : 2;
  __REG32  WDogEnOv             : 1;
  __REG32                       : 8;
} __scctrl_bits;

/* SCIMCTRL register */
typedef struct
{
  __REG32  ItMdEn               : 1;
  __REG32  ItMdCtrl             : 3;
  __REG32                       : 3;
  __REG32  InMdType             : 1;
  __REG32                       :24;
} __scimctrl_bits;

/* SCIMSTAT register */
typedef struct
{
  __REG32  ItMdStat             : 1;
  __REG32                       :31;
} __scimstat_bits;

/* SCXTALCTRL register */
typedef struct
{
  __REG32  XtalOver             : 1;
  __REG32  XtalEn               : 1;
  __REG32  XtalStat             : 1;
  __REG32  XtalTime             :16;
  __REG32                       :13;
} __scxtalctrl_bits;

/* SCPLLCTRL register */
typedef struct
{
  __REG32  PllOver              : 1;
  __REG32  PllEn                : 1;
  __REG32  PllStat              : 1;
  __REG32  PllTime              :25;
  __REG32                       : 4;
} __scpllctrl_bits;

/* SMI_CR1 register */
typedef struct
{
  __REG32  BE                   : 4;
  __REG32  TCS                  : 4;
  __REG32  PRESC                : 7;
  __REG32  FAST                 : 1;
  __REG32  HOLD                 : 8;
  __REG32  ADD_LENGTH           : 4;
  __REG32  SW                   : 1;
  __REG32  WBM                  : 1;
  __REG32                       : 2;
} __smi_cr1_bits;

/* SMI_CR2 register */
typedef struct
{
  __REG32  TRA_LENGTH           : 3;
  __REG32                       : 1;
  __REG32  REC_LENGTH           : 3;
  __REG32  SEND                 : 1;
  __REG32  TFIE                 : 1;
  __REG32  WCIE                 : 1;
  __REG32  RSR                  : 1;
  __REG32  WEN                  : 1;
  __REG32  BS                   : 2;
  __REG32                       :18;
} __smi_cr2_bits;

/* SMI_SR register */
typedef struct
{
  __REG32  SMSR                 : 8;
  __REG32  TFF                  : 1;
  __REG32  WCF                  : 1;
  __REG32  ERF2                 : 1;
  __REG32  ERF1                 : 1;
  __REG32  WM                   : 4;
  __REG32                       :16;
} __smi_sr_bits;

/* WdogControl register */
typedef struct
{
  __REG32  INTEN                : 1;
  __REG32  RESEN                : 1;
  __REG32                       :30;
} __wdogcontrol_bits;

/* WdogRIS register */
typedef struct
{
  __REG32  WDOGRIS              : 1;
  __REG32                       :31;
} __wdogris_bits;

/* WdogMIS register */
typedef struct
{
  __REG32  WDOGMIS              : 1;
  __REG32                       :31;
} __wdogmis_bits;

/* Timer_control register */
typedef struct
{
  __REG16  PRESCALER            : 4;
  __REG16  MODE                 : 1;
  __REG16  ENABLE               : 1;
  __REG16  CAPTURE              : 2;
  __REG16  MATCH_INT            : 1;
  __REG16  FEDGE_INT            : 1;
  __REG16  REDGE_INT            : 1;
  __REG16                       : 5;
} __timer_control_bits;

/* TIMER_STATUS_INT_ACK register */
typedef struct
{
  __REG16  MATCH                : 1;
  __REG16  FEDGE                : 1;
  __REG16  REDGE                : 1;
  __REG16                       :13;
} __timer_status_int_ack_bits;

/* GPIODIR register */
typedef struct
{
  __REG16  GPIODIR0             : 1;
  __REG16  GPIODIR1             : 1;
  __REG16  GPIODIR2             : 1;
  __REG16  GPIODIR3             : 1;
  __REG16  GPIODIR4             : 1;
  __REG16  GPIODIR5             : 1;
  __REG16                       :10;
} __gpiodir_bits;

/* GPIODATA register */
typedef struct
{
  __REG16  GPIODATA0            : 1;
  __REG16  GPIODATA1            : 1;
  __REG16  GPIODATA2            : 1;
  __REG16  GPIODATA3            : 1;
  __REG16  GPIODATA4            : 1;
  __REG16  GPIODATA5            : 1;
  __REG16  GPIODATA6            : 1;
  __REG16  GPIODATA7            : 1;
  __REG16                       : 8;
} __gpiodata_bits;

/* GPIOIS register */
typedef struct
{
  __REG16  GPIOIS0              : 1;
  __REG16  GPIOIS1              : 1;
  __REG16  GPIOIS2              : 1;
  __REG16  GPIOIS3              : 1;
  __REG16  GPIOIS4              : 1;
  __REG16  GPIOIS5              : 1;
  __REG16                       :10;
} __gpiois_bits;

/* GPIOIBE register */
typedef struct
{
  __REG16  GPIOIBE0             : 1;
  __REG16  GPIOIBE1             : 1;
  __REG16  GPIOIBE2             : 1;
  __REG16  GPIOIBE3             : 1;
  __REG16  GPIOIBE4             : 1;
  __REG16  GPIOIBE5             : 1;
  __REG16                       :10;
} __gpioibe_bits;

/* GPIOIEV register */
typedef struct
{
  __REG16  GPIOIEV0             : 1;
  __REG16  GPIOIEV1             : 1;
  __REG16  GPIOIEV2             : 1;
  __REG16  GPIOIEV3             : 1;
  __REG16  GPIOIEV4             : 1;
  __REG16  GPIOIEV5             : 1;
  __REG16                       :10;
} __gpioiev_bits;

/* GPIOIE register */
typedef struct
{
  __REG16  GPIOIE0              : 1;
  __REG16  GPIOIE1              : 1;
  __REG16  GPIOIE2              : 1;
  __REG16  GPIOIE3              : 1;
  __REG16  GPIOIE4              : 1;
  __REG16  GPIOIE5              : 1;
  __REG16                       :10;
} __gpioie_bits;

/* GPIORIS register */
typedef struct
{
  __REG16  GPIORIS0             : 1;
  __REG16  GPIORIS1             : 1;
  __REG16  GPIORIS2             : 1;
  __REG16  GPIORIS3             : 1;
  __REG16  GPIORIS4             : 1;
  __REG16  GPIORIS5             : 1;
  __REG16                       :10;
} __gpioris_bits;

/* GPIOMIS register */
typedef struct
{
  __REG16  GPIOMIS0             : 1;
  __REG16  GPIOMIS1             : 1;
  __REG16  GPIOMIS2             : 1;
  __REG16  GPIOMIS3             : 1;
  __REG16  GPIOMIS4             : 1;
  __REG16  GPIOMIS5             : 1;
  __REG16                       :10;
} __gpiomis_bits;

/* GPIOIC register */
typedef struct
{
  __REG16  GPIOIC0              : 1;
  __REG16  GPIOIC1              : 1;
  __REG16  GPIOIC2              : 1;
  __REG16  GPIOIC3              : 1;
  __REG16  GPIOIC4              : 1;
  __REG16  GPIOIC5              : 1;
  __REG16                       :10;
} __gpioic_bits;

/* DMACIntStatus register */
typedef struct
{
  __REG32  IntStatus0           : 1;
  __REG32  IntStatus1           : 1;
  __REG32  IntStatus2           : 1;
  __REG32  IntStatus3           : 1;
  __REG32  IntStatus4           : 1;
  __REG32  IntStatus5           : 1;
  __REG32  IntStatus6           : 1;
  __REG32  IntStatus7           : 1;
  __REG32                       :24;
} __dmacintstatus_bits;

/* DMACIntTCStatus register */
typedef struct
{
  __REG32  IntTCStatus0         : 1;
  __REG32  IntTCStatus1         : 1;
  __REG32  IntTCStatus2         : 1;
  __REG32  IntTCStatus3         : 1;
  __REG32  IntTCStatus4         : 1;
  __REG32  IntTCStatus5         : 1;
  __REG32  IntTCStatus6         : 1;
  __REG32  IntTCStatus7         : 1;
  __REG32                       :24;
} __dmacinttcstatus_bits;

/* DMACIntTCClear register */
typedef struct
{
  __REG32  IntTCClear0          : 1;
  __REG32  IntTCClear1          : 1;
  __REG32  IntTCClear2          : 1;
  __REG32  IntTCClear3          : 1;
  __REG32  IntTCClear4          : 1;
  __REG32  IntTCClear5          : 1;
  __REG32  IntTCClear6          : 1;
  __REG32  IntTCClear7          : 1;
  __REG32                       :24;
} __dmacinttcclear_bits;

/* DMACIntErrorStatus register */
typedef struct
{
  __REG32  IntErrorStatus0      : 1;
  __REG32  IntErrorStatus1      : 1;
  __REG32  IntErrorStatus2      : 1;
  __REG32  IntErrorStatus3      : 1;
  __REG32  IntErrorStatus4      : 1;
  __REG32  IntErrorStatus5      : 1;
  __REG32  IntErrorStatus6      : 1;
  __REG32  IntErrorStatus7      : 1;
  __REG32                       :24;
} __dmacinterrorstatus_bits;

/* DMACIntErrClr register */
typedef struct
{
  __REG32  IntErrClr0           : 1;
  __REG32  IntErrClr1           : 1;
  __REG32  IntErrClr2           : 1;
  __REG32  IntErrClr3           : 1;
  __REG32  IntErrClr4           : 1;
  __REG32  IntErrClr5           : 1;
  __REG32  IntErrClr6           : 1;
  __REG32  IntErrClr7           : 1;
  __REG32                       :24;
} __dmacinterrclr_bits;

/* DMACRawIntTCStatus register */
typedef struct
{
  __REG32  RawIntTCStatus0      : 1;
  __REG32  RawIntTCStatus1      : 1;
  __REG32  RawIntTCStatus2      : 1;
  __REG32  RawIntTCStatus3      : 1;
  __REG32  RawIntTCStatus4      : 1;
  __REG32  RawIntTCStatus5      : 1;
  __REG32  RawIntTCStatus6      : 1;
  __REG32  RawIntTCStatus7      : 1;
  __REG32                       :24;
} __dmacrawinttcstatus_bits;

/* DMACRawIntErrorStatus register */
typedef struct
{
  __REG32  RawIntErrorStatus0   : 1;
  __REG32  RawIntErrorStatus1   : 1;
  __REG32  RawIntErrorStatus2   : 1;
  __REG32  RawIntErrorStatus3   : 1;
  __REG32  RawIntErrorStatus4   : 1;
  __REG32  RawIntErrorStatus5   : 1;
  __REG32  RawIntErrorStatus6   : 1;
  __REG32  RawIntErrorStatus7   : 1;
  __REG32                       :24;
} __dmacrawinterrorstatus_bits;

/* DMACEnbldChns register */
typedef struct
{
  __REG32  EnabledChannels0     : 1;
  __REG32  EnabledChannels1     : 1;
  __REG32  EnabledChannels2     : 1;
  __REG32  EnabledChannels3     : 1;
  __REG32  EnabledChannels4     : 1;
  __REG32  EnabledChannels5     : 1;
  __REG32  EnabledChannels6     : 1;
  __REG32  EnabledChannels7     : 1;
  __REG32                       :24;
} __dmacenbldchns_bits;

/* DMACSoftBReq register */
typedef struct
{
  __REG32  SoftBReq0            : 1;
  __REG32  SoftBReq1            : 1;
  __REG32  SoftBReq2            : 1;
  __REG32  SoftBReq3            : 1;
  __REG32  SoftBReq4            : 1;
  __REG32  SoftBReq5            : 1;
  __REG32  SoftBReq6            : 1;
  __REG32  SoftBReq7            : 1;
  __REG32  SoftBReq8            : 1;
  __REG32  SoftBReq9            : 1;
  __REG32  SoftBReq10           : 1;
  __REG32  SoftBReq11           : 1;
  __REG32  SoftBReq12           : 1;
  __REG32  SoftBReq13           : 1;
  __REG32  SoftBReq14           : 1;
  __REG32  SoftBReq15           : 1;
  __REG32                       :16;
} __dmacsoftbreq_bits;

/* DMACSoftSReq register */
typedef struct
{
  __REG32  SoftSReq0            : 1;
  __REG32  SoftSReq1            : 1;
  __REG32  SoftSReq2            : 1;
  __REG32  SoftSReq3            : 1;
  __REG32  SoftSReq4            : 1;
  __REG32  SoftSReq5            : 1;
  __REG32  SoftSReq6            : 1;
  __REG32  SoftSReq7            : 1;
  __REG32  SoftSReq8            : 1;
  __REG32  SoftSReq9            : 1;
  __REG32  SoftSReq10           : 1;
  __REG32  SoftSReq11           : 1;
  __REG32  SoftSReq12           : 1;
  __REG32  SoftSReq13           : 1;
  __REG32  SoftSReq14           : 1;
  __REG32  SoftSReq15           : 1;
  __REG32                       :16;
} __dmacsoftsreq_bits;

/* DMACSoftLBReq register */
typedef struct
{
  __REG32  SoftLBReq0           : 1;
  __REG32  SoftLBReq1           : 1;
  __REG32  SoftLBReq2           : 1;
  __REG32  SoftLBReq3           : 1;
  __REG32  SoftLBReq4           : 1;
  __REG32  SoftLBReq5           : 1;
  __REG32  SoftLBReq6           : 1;
  __REG32  SoftLBReq7           : 1;
  __REG32  SoftLBReq8           : 1;
  __REG32  SoftLBReq9           : 1;
  __REG32  SoftLBReq10          : 1;
  __REG32  SoftLBReq11          : 1;
  __REG32  SoftLBReq12          : 1;
  __REG32  SoftLBReq13          : 1;
  __REG32  SoftLBReq14          : 1;
  __REG32  SoftLBReq15          : 1;
  __REG32                       :16;
} __dmacsoftlbreq_bits;

/* DMACSoftLSReq register */
typedef struct
{
  __REG32  SoftLSReq0           : 1;
  __REG32  SoftLSReq1           : 1;
  __REG32  SoftLSReq2           : 1;
  __REG32  SoftLSReq3           : 1;
  __REG32  SoftLSReq4           : 1;
  __REG32  SoftLSReq5           : 1;
  __REG32  SoftLSReq6           : 1;
  __REG32  SoftLSReq7           : 1;
  __REG32  SoftLSReq8           : 1;
  __REG32  SoftLSReq9           : 1;
  __REG32  SoftLSReq10          : 1;
  __REG32  SoftLSReq11          : 1;
  __REG32  SoftLSReq12          : 1;
  __REG32  SoftLSReq13          : 1;
  __REG32  SoftLSReq14          : 1;
  __REG32  SoftLSReq15          : 1;
  __REG32                       :16;
} __dmacsoftlsreq_bits;

/* DMAC configuration register */
typedef struct
{
  __REG32  E                    : 1;
  __REG32  M1                   : 1;
  __REG32  M2                   : 1;
  __REG32                       :29;
} __dmacconfiguration_bits;

/* DMACSoftLSReq register */
typedef struct
{
  __REG32  DMACSync0            : 1;
  __REG32  DMACSync1            : 1;
  __REG32  DMACSync2            : 1;
  __REG32  DMACSync3            : 1;
  __REG32  DMACSync4            : 1;
  __REG32  DMACSync5            : 1;
  __REG32  DMACSync6            : 1;
  __REG32  DMACSync7            : 1;
  __REG32  DMACSync8            : 1;
  __REG32  DMACSync9            : 1;
  __REG32  DMACSync10           : 1;
  __REG32  DMACSync11           : 1;
  __REG32  DMACSync12           : 1;
  __REG32  DMACSync13           : 1;
  __REG32  DMACSync14           : 1;
  __REG32  DMACSync15           : 1;
  __REG32                       :16;
} __dmacsync_bits;

/* DMACCnLLI register */
typedef struct
{
  __REG32  LM                   : 1;
  __REG32                       : 1;
  __REG32  LLI                  :30;
} __dmacclli_bits;

/* DMACCn control register */
typedef struct
{
  __REG32  TS                   :12;
  __REG32  SBSize               : 3;
  __REG32  DBSize               : 3;
  __REG32  Swidth               : 3;
  __REG32  Dwidth               : 3;
  __REG32  S                    : 1;
  __REG32  D                    : 1;
  __REG32  SI                   : 1;
  __REG32  DI                   : 1;
  __REG32  Port                 : 3;
  __REG32  I                    : 1;
} __dmacccontrol_bits;

/* DMAC Configuration register */
typedef struct
{
  __REG32  E                    : 1;
  __REG32  SrcPeripheral        : 4;
  __REG32                       : 1;
  __REG32  DestPeripheral       : 4;
  __REG32                       : 1;
  __REG32  FlowCntrl            : 3;
  __REG32  IE                   : 1;
  __REG32  ITC                  : 1;
  __REG32  L                    : 1;
  __REG32  A                    : 1;
  __REG32  H                    : 1;
  __REG32                       :13;
} __dmaccconfiguration_bits;

/* CONTROL register */
typedef struct
{
  __REG32  MASK                 : 6;
  __REG32                       : 2;
  __REG32  PB                   : 1;
  __REG32  TB                   : 1;
  __REG32                       :21;
  __REG32  IE                   : 1;
} __rtccontrol_bits;

/* STATUS register */
typedef struct
{
  __REG32  RC                   : 1;
  __REG32                       : 1;
  __REG32  PT                   : 1;
  __REG32  PD                   : 1;
  __REG32  LT                   : 1;
  __REG32  LD                   : 1;
  __REG32                       :25;
  __REG32  I                    : 1;
} __rtcstatus_bits;

/* TIME register */
typedef struct
{
  __REG32  SU                   : 4;
  __REG32  ST                   : 3;
  __REG32                       : 1;
  __REG32  MU                   : 4;
  __REG32  MT                   : 3;
  __REG32                       : 1;
  __REG32  HU                   : 4;
  __REG32  HT                   : 2;
  __REG32                       :10;
} __rtctime_bits;

/* DATE register */
typedef struct
{
  __REG32  DU                   : 4;
  __REG32  DT                   : 2;
  __REG32                       : 2;
  __REG32  MU                   : 4;
  __REG32  MT                   : 3;
  __REG32                       : 1;
  __REG32  YU                   : 4;
  __REG32  YT                   : 4;
  __REG32  YH                   : 4;
  __REG32  YM                   : 4;
} __rtcdate_bits;

#if 0
/* Status and control register (SYS_SCR) */
typedef struct
{
  __REG32  C0SL                 : 1;
  __REG32  C0SH                 : 1;
  __REG32  C1SL                 : 1;
  __REG32  C1SH                 : 1;
  __REG32  C2SL                 : 1;
  __REG32  C2SH                 : 1;
  __REG32  C3SL                 : 1;
  __REG32  C3SH                 : 1;
  __REG32  C4SL                 : 1;
  __REG32  C4SH                 : 1;
  __REG32  C5SL                 : 1;
  __REG32  C5SH                 : 1;
  __REG32  C6SL                 : 1;
  __REG32  C6SH                 : 1;
  __REG32  C7SL                 : 1;
  __REG32  C7SH                 : 1;
  __REG32  C8SL                 : 1;
  __REG32  C8SH                 : 1;
  __REG32  C9SL                 : 1;
  __REG32  C9SH                 : 1;
  __REG32  C10SL                : 1;
  __REG32  C10SH                : 1;
  __REG32  C11SL                : 1;
  __REG32  C11SH                : 1;
  __REG32  C12SL                : 1;
  __REG32  C12SH                : 1;
  __REG32  C13SL                : 1;
  __REG32  C13SH                : 1;
  __REG32  C14SL                : 1;
  __REG32  C14SH                : 1;
  __REG32  C15SL                : 1;
  __REG32  C15SH                : 1;
} __c3_sys_scr_bits;

/* Hardware Version and Revision Registers (SYS_VER) */
typedef struct
{
  __REG32  S                    :16;
  __REG32  R                    : 8;
  __REG32  V                    : 8;
} __c3_sys_ver_bits;

/* Memory size register (HIF_MSIZE) */
typedef struct
{
  __REG32  S                    :17;
  __REG32                       :15;
} __c3_hif_msize_bits;

/* Memory control register (HIF_MCAR) */
typedef struct
{
  __REG32  BMM                  : 1;
  __REG32                       :15;
  __REG32  DAIW                 : 1;
  __REG32  DAIR                 : 1;
  __REG32                       :14;
} __c3_hif_mcr_bits;

/* Memory access address register (HIF_MAAR) */
typedef struct
{
  __REG32  A                    :16;
  __REG32  B                    :16;
} __c3_hif_maar_bits;

/* Byte bucket control register (HIF_NCR) */
typedef struct
{
  __REG32  ENM                  : 1;
  __REG32                       :31;
} __c3_hif_ncr_bits;

/* Status and Control Register (ID_SCR) */
typedef struct
{
  __REG32  C0SL                 : 1;
  __REG32  C0SH                 : 1;
  __REG32  C1SL                 : 1;
  __REG32  C1SH                 : 1;
  __REG32  C2SL                 : 1;
  __REG32  C2SH                 : 1;
  __REG32  C3SL                 : 1;
  __REG32  C3SH                 : 1;
  __REG32  C4SL                 : 1;
  __REG32  C4SH                 : 1;
  __REG32  C5SL                 : 1;
  __REG32  C5SH                 : 1;
  __REG32  C6SL                 : 1;
  __REG32  C6SH                 : 1;
  __REG32  C7SL                 : 1;
  __REG32  C7SH                 : 1;
  __REG32  RST                  : 1;
  __REG32  IGR                  : 1;
  __REG32                       : 1;
  __REG32  SSE                  : 1;
  __REG32  SSC                  : 1;
  __REG32  IER                  : 1;
  __REG32  IES                  : 1;
  __REG32  IS                   : 1;
  __REG32  CDNX                 : 1;
  __REG32  CBSY                 : 1;
  __REG32  CERR                 : 1;
  __REG32                       : 2;
  __REG32  BERR                 : 1;
  __REG32  IDSL                 : 1;
  __REG32  IDSH                 : 1;
} __c3_id_scr_bits;

#endif

/* Control register */
typedef struct
{
  __REG8  PeripSize            : 2;
  __REG8  ByteEnWrite          : 1;
  __REG8  ByteEnRead           : 1;
  __REG8                       : 4;
} __control_reg_bits;

/* Control register */
typedef struct
{
  __REG8  CS0                  : 1;
  __REG8  CS1                  : 1;
  __REG8  CS2                  : 1;
  __REG8  CS3                  : 1;
  __REG8  CS4                  : 1;
  __REG8  CS5                  : 1;
  __REG8                       : 2;
} __ack_reg_bits;

/* IRQ register */
typedef struct
{
  __REG8  SizeMismatch         : 1;
  __REG8  TimeoutError         : 1;
  __REG8  BusAccessError       : 1;
  __REG8                       : 5;
} __irq_reg_bits;

/* HCCAPBASE register */
typedef struct
{
  __REG32  CAPLENGTH            : 8;
  __REG32                       : 8;
  __REG32  HCIVERSION           :16;
} __hccapbase_bits;

/* HCSPARAMS register */
typedef struct
{
  __REG32  N_PORTS              : 4;
  __REG32  PPC                  : 1;
  __REG32                       : 2;
  __REG32  PRR                  : 1;
  __REG32  N_PCC                : 4;
  __REG32  N_CC                 : 4;
  __REG32  P_INDICATOR          : 1;
  __REG32                       : 3;
  __REG32  DPN                  : 4;
  __REG32                       : 8;
} __hcsparams_bits;

/* HCCPARAMS register */
typedef struct
{
  __REG32  _64BAC               : 1;
  __REG32  PFLF                 : 1;
  __REG32  ASPC                 : 1;
  __REG32                       : 1;
  __REG32  IST                  : 4;
  __REG32  EECP                 : 8;
  __REG32                       :16;
} __hccparams_bits;

/* USBCMD register */
typedef struct
{
  __REG32  RS                   : 1;
  __REG32  HCRESET              : 1;
  __REG32  FLS                  : 2;
  __REG32  PSE                  : 1;
  __REG32  ASE                  : 1;
  __REG32  IAAD                 : 1;
  __REG32  LHCR                 : 1;
  __REG32  ASPMC                : 2;
  __REG32                       : 1;
  __REG32  ASPME                : 1;
  __REG32                       : 4;
  __REG32  ITC                  : 8;
  __REG32                       : 8;
} __hcusbcmd_bits;

/* USBSTS register */
typedef struct
{
  __REG32  USBINT               : 1;
  __REG32  USBERRINT            : 1;
  __REG32  PCD                  : 1;
  __REG32  FLR                  : 1;
  __REG32  HSE                  : 1;
  __REG32  IAA                  : 1;
  __REG32                       : 6;
  __REG32  HH                   : 1;
  __REG32  R                    : 1;
  __REG32  PSS                  : 1;
  __REG32  ASS                  : 1;
  __REG32                       :16;
} __hcusbsts_bits;

/* USBINTR register */
typedef struct
{
  __REG32  IE                   : 1;
  __REG32  EIE                  : 1;
  __REG32  PCIE                 : 1;
  __REG32  FLRE                 : 1;
  __REG32  HSEE                 : 1;
  __REG32  IAAE                 : 1;
  __REG32                       :26;
} __hcusbintr_bits;

/* FRINDEX register */
typedef struct
{
  __REG32  FRAME                :14;
  __REG32                       :18;
} __hcfrindex_bits;

/* CONFIGFLAG register */
typedef struct
{
  __REG32  CF                   : 1;
  __REG32                       :31;
} __hcconfigflag_bits;

/* PORTSC registers */
typedef struct
{
  __REG32  CCS                  : 1;
  __REG32  CSC                  : 1;
  __REG32  PEN                  : 1;
  __REG32  PEDC                 : 1;
  __REG32  OcA                  : 1;
  __REG32  OcC                  : 1;
  __REG32  FPR                  : 1;
  __REG32  S                    : 1;
  __REG32  PR                   : 1;
  __REG32                       : 1;
  __REG32  LS                   : 2;
  __REG32  PP                   : 1;
  __REG32  PO                   : 1;
  __REG32  PIC                  : 2;
  __REG32  PTC                  : 4;
  __REG32  WKCNNT_E             : 1;
  __REG32  WKDSCNNT_E           : 1;
  __REG32  WKOC_E               : 1;
  __REG32                       : 9;
} __hcportsc_bits;

/* INSNREG00 register */
typedef struct
{
  __REG32  RMFS                 :14;
  __REG32                       :18;
} __hcinsnreg00_bits;

/* INSNREG01 register */
typedef struct
{
  __REG32  IN                   :16;
  __REG32  OUT                  :16;
} __hcinsnreg01_bits;

/* INSNREG02 register */
typedef struct
{
  __REG32  DEEP                 :12;
  __REG32                       :20;
} __hcinsnreg02_bits;

/* INSNREG03 register */
typedef struct
{
  __REG32  BMT                  : 1;
  __REG32                       :31;
} __hcinsnreg03_bits;

/* INSNREG05 register */
typedef struct
{
  __REG32  VStatus              : 8;
  __REG32  VControl             : 4;
  __REG32  VControlLoadM        : 1;
  __REG32  VPort                : 4;
  __REG32  VBusy                : 1;
  __REG32                       :14;
} __hcinsnreg05_bits;

/* HcRevision Register */
typedef struct {
  __REG32 REV               : 8;
  __REG32                   :24;
} __hcrevision_bits;

/* HcControl Register */
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

/* HcCommandStatus Register */
typedef struct {
  __REG32 HCR               : 1;
  __REG32 CLF               : 1;
  __REG32 BLF               : 1;
  __REG32 OCR               : 1;
  __REG32                   :12;
  __REG32 SOC               : 2;
  __REG32                   :14;
} __hccommandstatus_bits;

/* HcInterruptStatus Register */
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

/* HcInterruptEnable Register
   HcInterruptDisable Register */
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

/* HcHCCA Register */
typedef struct {
  __REG32                   : 8;
  __REG32 HCCA              :24;
} __hchcca_bits;

/* HcPeriodCurrentED Register */
typedef struct {
  __REG32                   : 4;
  __REG32 PCED              :28;
} __hcperiodcurrented_bits;

/* HcControlHeadED Registerr */
typedef struct {
  __REG32                   : 4;
  __REG32 CHED              :28;
} __hccontrolheaded_bits;

/* HcControlCurrentED Register */
typedef struct {
  __REG32                   : 4;
  __REG32 CCED              :28;
} __hccontrolcurrented_bits;

/* HcBulkHeadED Register */
typedef struct {
  __REG32                   : 4;
  __REG32 BHED              :28;
} __hcbulkheaded_bits;

/* HcBulkCurrentED Register */
typedef struct {
  __REG32                   : 4;
  __REG32 BCED              :28;
} __hcbulkcurrented_bits;

/* HcDoneHead Register */
typedef struct {
  __REG32                   : 4;
  __REG32 DH                :28;
} __hcdonehead_bits;

/* HcFmInterval Register */
typedef struct {
  __REG32 FI                :14;
  __REG32                   : 2;
  __REG32 FSMPS             :15;
  __REG32 FIT               : 1;
} __hcfminterval_bits;

/* HcFmRemaining Register */
typedef struct {
  __REG32 FR                :14;
  __REG32                   :17;
  __REG32 FRT               : 1;
} __hcfmremaining_bits;

/* HcFmNumber Register */
typedef struct {
  __REG32 FN                :16;
  __REG32                   :16;
} __hcfmnumber_bits;

/* HcPeriodicStart Register */
typedef struct {
  __REG32 PS                :14;
  __REG32                   :18;
} __hcperiodicstart_bits;

/* HcLSThreshold Register */
typedef struct {
  __REG32 LST               :12;
  __REG32                   :20;
} __hclsthreshold_bits;

/* HcRhDescriptorA Register */
typedef struct {
  __REG32 NDP               : 8;
  __REG32 PSM               : 1;
  __REG32 NPS               : 1;
  __REG32 DT                : 1;
  __REG32 OCPM              : 1;
  __REG32 NOCP              : 1;
  __REG32                   :11;
  __REG32 POTPGT            : 8;
} __hcrhdescriptora_bits;

/* HcRhDescriptorB Register */
typedef struct {
  __REG32 DR                :16;
  __REG32 PPCM              :16;
} __hcrhdescriptorb_bits;

/* HcRhStatus Register */
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

/* HcRhPortStatus[1] Register */
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
  __REG32 OCIC              : 1;
  __REG32 PRSC              : 1;
  __REG32                   :11;
} __hcrhportstatus_bits;

/* Device configuration register */
typedef struct {
  __REG32 SPD               : 2;
  __REG32 RWKP              : 1;
  __REG32 SP                : 1;
  __REG32 SS                : 1;
  __REG32 PI                : 1;
  __REG32 DIR               : 1;
  __REG32 STATUS            : 1;
  __REG32 STATUS_1          : 1;
  __REG32 PHY_ERROR_DETECT  : 1;
  __REG32 FS_TIMEOUT_CALIB  : 3;
  __REG32 HS_TIMEOUT_CALIB  : 3;
  __REG32 HALT_STATUS       : 1;
  __REG32 CSR_PRG           : 1;
  __REG32 SET_DESC          : 1;
  __REG32                   :13;
} __uddcfg_bits;

/* Device configuration register */
typedef struct {
  __REG32 RES               : 1;
  __REG32                   : 1;
  __REG32 RDE               : 1;
  __REG32 TDE               : 1;
  __REG32 DU                : 1;
  __REG32 BE                : 1;
  __REG32 BF                : 1;
  __REG32 THE               : 1;
  __REG32 BREN              : 1;
  __REG32 MODE              : 1;
  __REG32 SD                : 1;
  __REG32 SCALE             : 1;
  __REG32 DEVNAK            : 1;
  __REG32 CSR_DONE          : 1;
  __REG32                   : 2;
  __REG32 BRLEN             : 8;
  __REG32 THLEN             : 8;
} __uddctrl_bits;

/* Device status register */
typedef struct {
  __REG32 CFG               : 4;
  __REG32 INTF              : 4;
  __REG32 ALT               : 4;
  __REG32 SUSP              : 1;
  __REG32 ENUM_SPD          : 2;
  __REG32 RXFIFO_EMPTY      : 1;
  __REG32 PHY_ERROR         : 1;
  __REG32                   : 1;
  __REG32 TS                :14;
} __uddstat_bits;

/* Device interrupt register */
typedef struct {
  __REG32 SC                : 1;
  __REG32 SI                : 1;
  __REG32 ES                : 1;
  __REG32 UR                : 1;
  __REG32 US                : 1;
  __REG32 SOF               : 1;
  __REG32 ENUM              : 1;
  __REG32                   :25;
} __uddintr_bits;

/* Device interrupt register */
typedef struct {
  __REG32 IN_EP0            : 1;
  __REG32 IN_EP1            : 1;
  __REG32 IN_EP2            : 1;
  __REG32 IN_EP3            : 1;
  __REG32 IN_EP4            : 1;
  __REG32 IN_EP5            : 1;
  __REG32 IN_EP6            : 1;
  __REG32 IN_EP7            : 1;
  __REG32 IN_EP8            : 1;
  __REG32 IN_EP9            : 1;
  __REG32 IN_EP10           : 1;
  __REG32 IN_EP11           : 1;
  __REG32 IN_EP12           : 1;
  __REG32 IN_EP13           : 1;
  __REG32 IN_EP14           : 1;
  __REG32 IN_EP15           : 1;
  __REG32 OUT_EP0           : 1;
  __REG32 OUT_EP1           : 1;
  __REG32 OUT_EP2           : 1;
  __REG32 OUT_EP3           : 1;
  __REG32 OUT_EP4           : 1;
  __REG32 OUT_EP5           : 1;
  __REG32 OUT_EP6           : 1;
  __REG32 OUT_EP7           : 1;
  __REG32 OUT_EP8           : 1;
  __REG32 OUT_EP9           : 1;
  __REG32 OUT_EP10          : 1;
  __REG32 OUT_EP11          : 1;
  __REG32 OUT_EP12          : 1;
  __REG32 OUT_EP13          : 1;
  __REG32 OUT_EP14          : 1;
  __REG32 OUT_EP15          : 1;
} __udeintr_bits;

/* Endpoint control register */
typedef struct {
  __REG32 S                 : 1;
  __REG32 F                 : 1;
  __REG32 SN                : 1;
  __REG32 P                 : 1;
  __REG32 ET                : 2;
  __REG32 NAK               : 1;
  __REG32 SNAK              : 1;
  __REG32 CNAK              : 1;
  __REG32 RRDY              : 1;
  __REG32                   :22;
} __udepctrl_bits;

/* Endpoint status register */
typedef struct {
  __REG32                   : 4;
  __REG32 OUT               : 2;
  __REG32 IN                : 1;
  __REG32 BNA               : 1;
  __REG32                   : 1;
  __REG32 HE                : 1;
  __REG32 TDC               : 1;
  __REG32 RX_PKT_SIZE       :12;
  __REG32 ISOIN_DONE        : 1;
  __REG32                   : 8;
} __udepstat_bits;

/* Endpoint buffer size and received packet frame number register */
typedef struct {
  __REG32 BUFF_SIZE         :16;
  __REG32 ISO_PID           : 2;
  __REG32                   :14;
} __udepbs_bits;

/* Endpoint maximum packet size and buffer size register */
typedef struct {
  __REG32 MAX_PKT_SIZE      :16;
  __REG32 BUFF_SIZE         :16;
} __udepmps_bits;

/* UDC20 endpoint register */
typedef struct {
  __REG32 EPNumber          : 4;
  __REG32 EPDir             : 1;
  __REG32 EPType            : 2;
  __REG32 ConfNumber        : 4;
  __REG32 InterfNumber      : 4;
  __REG32 AltSetting        : 4;
  __REG32 MaxPackSize       :11;
  __REG32                   : 2;
} __udep_bits;

/* Bus mode register */
typedef struct {
  __REG32 SWR               : 1;
  __REG32 DA                : 1;
  __REG32 DSL               : 5;
  __REG32                   : 1;
  __REG32 PBL               : 6;
  __REG32 PR                : 2;
  __REG32 FB                : 1;
  __REG32                   :15;
} __dma_bmr_bits;

/* Status register */
typedef struct {
  __REG32 TI                : 1;
  __REG32 TPS               : 1;
  __REG32 TU                : 1;
  __REG32 TJT               : 1;
  __REG32 OVF               : 1;
  __REG32 UNF               : 1;
  __REG32 RI                : 1;
  __REG32 RU                : 1;
  __REG32 RPS               : 1;
  __REG32 RWT               : 1;
  __REG32 ETI               : 1;
  __REG32                   : 2;
  __REG32 FBI               : 1;
  __REG32 ERI               : 1;
  __REG32 AIS               : 1;
  __REG32 NIS               : 1;
  __REG32 RS                : 3;
  __REG32 TS                : 3;
  __REG32 EB                : 3;
  __REG32                   : 1;
  __REG32 GMI               : 1;
  __REG32 GPI               : 1;
  __REG32                   : 3;
} __dma_sr_bits;

/* Status register */
typedef struct {
  __REG32                   : 1;
  __REG32 SR                : 1;
  __REG32 OSF               : 1;
  __REG32 RTC               : 2;
  __REG32                   : 1;
  __REG32 FUF               : 1;
  __REG32 FEF               : 1;
  __REG32 EFC               : 1;
  __REG32 RFA               : 2;
  __REG32 RFD               : 2;
  __REG32 ST                : 1;
  __REG32 TTC               : 3;
  __REG32                   : 3;
  __REG32 FTF               : 1;
  __REG32                   :11;
} __dma_omr_bits;

/* Interrupt enable register */
typedef struct {
  __REG32 TIE               : 1;
  __REG32 TSE               : 1;
  __REG32 TUE               : 1;
  __REG32 TJE               : 1;
  __REG32 OVE               : 1;
  __REG32 UNE               : 1;
  __REG32 RIE               : 1;
  __REG32 RUE               : 1;
  __REG32 RSE               : 1;
  __REG32 RWE               : 1;
  __REG32 ETE               : 1;
  __REG32                   : 2;
  __REG32 FBE               : 1;
  __REG32 ERE               : 1;
  __REG32 AIE               : 1;
  __REG32 NIE               : 1;
  __REG32                   :15;
} __dma_ier_bits;

/* Missed frame and buffer overflow counter register */
typedef struct {
  __REG32 NOFMBC            :16;
  __REG32 OFMFC             : 1;
  __REG32 NOFMBA            :11;
  __REG32 OFFIFOOC          : 1;
  __REG32                   : 3;
} __dma_mfabocr_bits;

/* MAC configuration register */
typedef struct {
  __REG32                   : 2;
  __REG32 RE                : 1;
  __REG32 TE                : 1;
  __REG32 DC                : 1;
  __REG32 BL                : 2;
  __REG32 ACS               : 1;
  __REG32                   : 1;
  __REG32 DR                : 1;
  __REG32 IPC               : 1;
  __REG32 DM                : 1;
  __REG32 LM                : 1;
  __REG32 DO                : 1;
  __REG32 FES               : 1;
  __REG32                   : 2;
  __REG32 IFG               : 3;
  __REG32 JE                : 1;
  __REG32                   : 1;
  __REG32 JD                : 1;
  __REG32 WD                : 1;
  __REG32                   : 8;
} __mac_cr_bits;

/* MAC frame filter register */
typedef struct {
  __REG32 PR                : 1;
  __REG32 HUC               : 1;
  __REG32 HMC               : 1;
  __REG32 DAIF              : 1;
  __REG32 PM                : 1;
  __REG32 DBF               : 1;
  __REG32 PCF               : 2;
  __REG32 SAIF              : 1;
  __REG32 SAF               : 1;
  __REG32                   :21;
  __REG32 RA                : 1;
} __mac_ffr_bits;

/* MII address register */
typedef struct {
  __REG32 GB                : 1;
  __REG32 GW                : 1;
  __REG32 CR                : 3;
  __REG32                   : 1;
  __REG32 GR                : 5;
  __REG32 PA                : 5;
  __REG32                   :16;
} __mac_miiar_bits;

/* MII data register */
typedef struct {
  __REG32 GD                :16;
  __REG32                   :16;
} __mac_miidr_bits;

/* Flow control register */
typedef struct {
  __REG32 FCB_BPA           : 1;
  __REG32 TFE               : 1;
  __REG32 RFE               : 1;
  __REG32 UP                : 1;
  __REG32 PLT               : 2;
  __REG32                   :10;
  __REG32 PT                :16;
} __mac_fcr_bits;

/* VLAN tag register */
typedef struct {
  __REG32 VL                :16;
  __REG32                   :16;
} __mac_vlantr_bits;

/* Version Register (RO) */
typedef struct {
  __REG32 VER               : 8;
  __REG32                   :24;
} __mac_vr_bits;

/* PMT control and status register */
typedef struct {
  __REG32 PD                : 1;
  __REG32 MPE               : 1;
  __REG32 WUFE              : 1;
  __REG32                   : 2;
  __REG32 MPR               : 1;
  __REG32 WUFR              : 1;
  __REG32                   : 2;
  __REG32 GU                : 1;
  __REG32                   :21;
  __REG32 WUFFPR            : 1;
} __mac_pmtcasr_bits;

/* Interrupt status register */
typedef struct {
  __REG32                   : 3;
  __REG32 PMTIS             : 1;
  __REG32 MMCIS             : 1;
  __REG32                   :27;
} __mac_ir_bits;

/* Interrupt mask register */
typedef struct {
  __REG32                   : 3;
  __REG32 PMTIM             : 1;
  __REG32                   :28;
} __mac_imr_bits;

/* MAC address0 high register */
typedef struct {
  __REG32 A                 :16;
  __REG32                   :15;
  __REG32 MO                : 1;
} __mac_ahr0_bits;

/* MAC address1-15 high register */
typedef struct {
  __REG32 A                 :16;
  __REG32                   : 8;
  __REG32 MBC               : 6;
  __REG32 SA                : 1;
  __REG32 AE                : 1;
} __mac_ahr_bits;

/* MMC control register */
typedef struct {
  __REG32 CR                : 1;
  __REG32 CSR               : 1;
  __REG32 ROR               : 1;
  __REG32                   :29;
} __mmc_cntrl_bits;

/* MMC receive interrupt register */
typedef struct {
  __REG32 FCGBH             : 1;
  __REG32 OCGBH             : 1;
  __REG32 OCGH              : 1;
  __REG32 BFCGH             : 1;
  __REG32 MFCGH             : 1;
  __REG32 CRCECH            : 1;
  __REG32 AECH              : 1;
  __REG32 RECH              : 1;
  __REG32 JECH              : 1;
  __REG32 USCGH             : 1;
  __REG32 OSCGH             : 1;
  __REG32 _63OCGBH          : 1;
  __REG32 _64T127OCGBH      : 1;
  __REG32 _128T255OCGBH     : 1;
  __REG32 _256T511OCGBH     : 1;
  __REG32 _512T1023OCGBH    : 1;
  __REG32 _1024OCGBH        : 1;
  __REG32 UFCGBH            : 1;
  __REG32 LECH              : 1;
  __REG32 OFTCH             : 1;
  __REG32 PFCH              : 1;
  __REG32 FIFOOCH           : 1;
  __REG32 VLANFCGBH         : 1;
  __REG32 WDECH             : 1;
  __REG32                   : 8;
} __mmc_intr_rx_bits;

/* MMC transmit interrupt register */
typedef struct {
  __REG32 OCGBH             : 1;
  __REG32 FCGBH             : 1;
  __REG32 BFCGH             : 1;
  __REG32 MFCGH             : 1;
  __REG32 _63OCGBH          : 1;
  __REG32 _64T127OCGBH      : 1;
  __REG32 _128T255OCGBH     : 1;
  __REG32 _256T511OCGBH     : 1;
  __REG32 _512T1023OCGBH    : 1;
  __REG32 _1024OCGBH        : 1;
  __REG32 UFCGBH            : 1;
  __REG32 MFCGBH            : 1;
  __REG32 BFCGBH            : 1;
  __REG32 UECH              : 1;
  __REG32 SCCGH             : 1;
  __REG32 MCCGH             : 1;
  __REG32 DCH               : 1;
  __REG32 LCCH              : 1;
  __REG32 ECCH              : 1;
  __REG32 CECH              : 1;
  __REG32 OCGH              : 1;
  __REG32 FCGH              : 1;
  __REG32 EDCH              : 1;
  __REG32 PFECH             : 1;
  __REG32 VLANFCGH          : 1;
  __REG32                   : 7;
} __mmc_intr_tx_bits;

/* MMC receive interrupt mask register */
typedef struct {
  __REG32 FCGBH             : 1;
  __REG32 OCGBH             : 1;
  __REG32 BFCGH             : 1;
  __REG32 MFCGH             : 1;
  __REG32 _63OCGBH          : 1;
  __REG32 _64T127OCGBH      : 1;
  __REG32 _128T255OCGBH     : 1;
  __REG32 _256T511OCGBH     : 1;
  __REG32 _512T1023OCGBH    : 1;
  __REG32 _1024OCGBH        : 1;
  __REG32 UFCGBH            : 1;
  __REG32 MFCGBH            : 1;
  __REG32 BFCGBH            : 1;
  __REG32 UECH              : 1;
  __REG32 SCCGH             : 1;
  __REG32 MCCGH             : 1;
  __REG32 DCH               : 1;
  __REG32 LCCH              : 1;
  __REG32 ECCH              : 1;
  __REG32 CECH              : 1;
  __REG32 PFCH              : 1;
  __REG32 FIFOOCH           : 1;
  __REG32 VLANFCGBH         : 1;
  __REG32 WDCH              : 1;
  __REG32                   : 8;
} __mmc_intr_mask_rx_bits;

/* MMC transmite interrupt mask register */
typedef struct {
  __REG32 OCGBH             : 1;
  __REG32 FCGBH             : 1;
  __REG32 BFCGH             : 1;
  __REG32 MFCGH             : 1;
  __REG32 _63OCGBH          : 1;
  __REG32 _64T127OCGBH      : 1;
  __REG32 _128T255OCGBH     : 1;
  __REG32 _256T511OCGBH     : 1;
  __REG32 _512T1023OCGBH    : 1;
  __REG32 _1024OCGBH        : 1;
  __REG32 UFCGBH            : 1;
  __REG32 MFCGBH            : 1;
  __REG32 BFCGBH            : 1;
  __REG32 UECH              : 1;
  __REG32 SCCGH             : 1;
  __REG32 MCCGH             : 1;
  __REG32 DCH               : 1;
  __REG32 LCCH              : 1;
  __REG32 ECCH              : 1;
  __REG32 CECH              : 1;
  __REG32 OCGH              : 1;
  __REG32 FCGH              : 1;
  __REG32 EDCH              : 1;
  __REG32 PFECH             : 1;
  __REG32 VLANFCGH          : 1;
  __REG32                   : 7;
} __mmc_intr_mask_tx_bits;

/* JPGCreg0 register */
typedef struct
{
  __REG32  StartStop            : 1;
  __REG32                       :31;
} __jpgcreg0_bits;

/* JPGCreg1 register */
typedef struct
{
  __REG32  Nf                   : 2;
  __REG32  Re                   : 1;
  __REG32  De                   : 1;
  __REG32  colspctype           : 2;
  __REG32  Ns                   : 2;
  __REG32  Hdr                  : 1;
  __REG32                       : 7;
  __REG32  Ysiz                 :16;
} __jpgcreg1_bits;

/* JPGCreg2 register */
typedef struct
{
  __REG32  NMCU                 :26;
  __REG32                       : 6;
} __jpgcreg2_bits;

/* JPGCreg3 register */
typedef struct
{
  __REG32  NRST                 :16;
  __REG32  Xsiz                 :16;
} __jpgcreg3_bits;

/* JPGCreg4-7 register */
typedef struct
{
  __REG32  HDi                  : 1;
  __REG32  HAi                  : 1;
  __REG32  QTi                  : 2;
  __REG32  Nblocki              : 4;
  __REG32  Hi                   : 4;
  __REG32  Vi                   : 4;
  __REG32                       :16;
} __jpgcreg4_bits;

/* JPGC control status register */
typedef struct
{
  __REG32  INT                  : 1;
  __REG32  BNV                  : 2;
  __REG32  LLI                  :15;
  __REG32                       :12;
  __REG32  SCR                  : 1;
  __REG32  EOC                  : 1;
} __jpgccs_bits;

/* JPGC bust count beforeInit register */
typedef struct
{
  __REG32  BTF                  :31;
  __REG32  EN                   : 1;
} __jpgcbcbi_bits;

/* IrDA_CON register */
typedef struct
{
  __REG32  RUN                  : 1;
  __REG32                       :31;
} __irda_con_bits;

/* IrDA_CONF register */
typedef struct
{
  __REG32  RATV                 :13;
  __REG32                       : 3;
  __REG32  BS                   : 3;
  __REG32  POLRX                : 1;
  __REG32  POLTX                : 1;
  __REG32                       :11;
} __irda_conf_bits;

/* IrDA_PARA register */
typedef struct
{
  __REG32  MODE                 : 2;
  __REG32  ABF                  : 6;
  __REG32                       : 8;
  __REG32  MNRB                 :12;
  __REG32                       : 4;
} __irda_para_bits;

/* IrDA_DV register */
typedef struct
{
  __REG32  N                    : 8;
  __REG32  INC                  : 8;
  __REG32  DEC                  :11;
  __REG32                       : 5;
} __irda_dv_bits;

/* IrDA_STAT register */
typedef struct
{
  __REG32  RXS                  : 1;
  __REG32  TXS                  : 1;
  __REG32                       :30;
} __irda_stat_bits;

/* IrDA_TFS register */
typedef struct
{
  __REG32  TFS                  :12;
  __REG32                       :20;
} __irda_tfs_bits;

/* IrDA_RFS register */
typedef struct
{
  __REG32  RFS                  :12;
  __REG32                       :20;
} __irda_rfs_bits;

/* IrDA_IMSC register */
typedef struct
{
  __REG32  LSREQ                : 1;
  __REG32  SREQ                 : 1;
  __REG32  LBREQ                : 1;
  __REG32  BREQ                 : 1;
  __REG32  FT                   : 1;
  __REG32  SD                   : 1;
  __REG32  FI                   : 1;
  __REG32  FD                   : 1;
  __REG32                       :24;
} __irda_imsc_bits;

/* IrDA_DMA register */
typedef struct
{
  __REG32  LSREQEN              : 1;
  __REG32  SREQEN               : 1;
  __REG32  LBREQEN              : 1;
  __REG32  BREQEN               : 1;
  __REG32                       :28;
} __irda_dma_bits;

/* UARTRSR register */
typedef struct
{
  __REG8  FE                   : 1;
  __REG8  PE                   : 1;
  __REG8  BE                   : 1;
  __REG8  OE                   : 1;
  __REG8                       : 4;
} __uartrsr_bits;

/* UARTFR register */
typedef struct
{
  __REG16  CTS                  : 1;
  __REG16  DSR                  : 1;
  __REG16  DCD                  : 1;
  __REG16  BUSY                 : 1;
  __REG16  RXFE                 : 1;
  __REG16  TXFF                 : 1;
  __REG16  RXFF                 : 1;
  __REG16  TXFE                 : 1;
  __REG16  RI                   : 1;
  __REG16                       : 7;
} __uartfr_bits;

/* UARTFBRD register */
typedef struct
{
  __REG8   DIVFRAC              : 6;
  __REG8                        : 2;
} __uartfbrd_bits;

/* UARTLCR_H register */
typedef struct
{
  __REG16  BRK                  : 1;
  __REG16  PEN                  : 1;
  __REG16  EPS                  : 1;
  __REG16  STP2                 : 1;
  __REG16  FEN                  : 1;
  __REG16  WLEN                 : 2;
  __REG16  SPS                  : 1;
  __REG16                       : 8;
} __uartlcr_h_bits;

/* UARTCR register */
typedef struct
{
  __REG16  UARTEN               : 1;
  __REG16                       : 6;
  __REG16  LBE                  : 1;
  __REG16  TXE                  : 1;
  __REG16  RXE                  : 1;
  __REG16  DTR                  : 1;
  __REG16  RTS                  : 1;
  __REG16  Out1                 : 1;
  __REG16  Out2                 : 1;
  __REG16  RTSEn                : 1;
  __REG16  CTSEn                : 1;
} __uartcr_bits;

/* UARTIFLS register */
typedef struct
{
  __REG16  TXIFLSEL             : 3;
  __REG16  RXIFLSEL             : 3;
  __REG16                       :10;
} __uartifls_bits;

/* UARTIMSC register */
typedef struct
{
  __REG16  RIMIM                : 1;
  __REG16  CTSMIM               : 1;
  __REG16  DCDMIM               : 1;
  __REG16  DSRMIM               : 1;
  __REG16  RXIM                 : 1;
  __REG16  TXIM                 : 1;
  __REG16  RTIM                 : 1;
  __REG16  FEIM                 : 1;
  __REG16  PEIM                 : 1;
  __REG16  BEIM                 : 1;
  __REG16  OEIM                 : 1;
  __REG16                       : 5;
} __uartimsc_bits;

/* UARTRIS register */
typedef struct
{
  __REG16  RIRMIS               : 1;
  __REG16  CTSRMIS              : 1;
  __REG16  DCDRMIS              : 1;
  __REG16  DSRRMIS              : 1;
  __REG16  RXRIS                : 1;
  __REG16  TXRIS                : 1;
  __REG16  RTRIS                : 1;
  __REG16  FERIS                : 1;
  __REG16  PERIS                : 1;
  __REG16  BERIS                : 1;
  __REG16  OERIS                : 1;
  __REG16                       : 5;
} __uartris_bits;

/* UARTMIS Register */
typedef struct
{
  __REG16  RIMMIS               : 1;
  __REG16  CTSMMIS              : 1;
  __REG16  DCDMMIS              : 1;
  __REG16  DSRMMIS              : 1;
  __REG16  RXMIS                : 1;
  __REG16  TXMIS                : 1;
  __REG16  RTMIS                : 1;
  __REG16  FEMIS                : 1;
  __REG16  PEMIS                : 1;
  __REG16  BEMIS                : 1;
  __REG16  OEMIS                : 1;
  __REG16                       : 5;
} __uartmis_bits;

/* UARTICR register */
typedef struct
{
  __REG16  RIMIC                : 1;
  __REG16  CTSMIC               : 1;
  __REG16  DCDMIC               : 1;
  __REG16  DSRMIC               : 1;
  __REG16  RXIC                 : 1;
  __REG16  TXIC                 : 1;
  __REG16  RTIC                 : 1;
  __REG16  FEIC                 : 1;
  __REG16  PEIC                 : 1;
  __REG16  BEIC                 : 1;
  __REG16  OEIC                 : 1;
  __REG16                       : 5;
} __uarticr_bits;

/* UARTDMACR register */
typedef struct
{
  __REG16  RXDMAE               : 1;
  __REG16  TXDMAE               : 1;
  __REG16  DMAONERR             : 1;
  __REG16                       :13;
} __uartdmacr_bits;

/* IC_CON register */
typedef struct
{
  __REG16  MASTER_MODE          : 1;
  __REG16  SPEED                : 2;
  __REG16  IC_10BITADDR_SLAVE   : 1;
  __REG16  IC_10BITADDR_MASTER  : 1;
  __REG16  IC_RESTART_EN        : 1;
  __REG16  IC_SLAVE_DISABLE     : 1;
  __REG16                       : 9;
} __ic_con_bits;

/* IC_TAR register */
typedef struct
{
  __REG16  IC_TAR               :10;
  __REG16  GC_OR_START          : 1;
  __REG16  SPECIAL              : 1;
  __REG16  IC_10BITADDR_MASTER  : 1;
  __REG16                       : 3;
} __ic_tar_bits;

/* IC_SAR register */
typedef struct
{
  __REG16  IC_SAR               :10;
  __REG16                       : 6;
} __ic_sar_bits;

/* IC_HS_MADDR register */
typedef struct
{
  __REG16  IC_HS_MAR            : 3;
  __REG16                       :13;
} __ic_hs_maddr_bits;

/* IC_DATA_CMD register */
typedef struct
{
  __REG16  DAT                  : 8;
  __REG16  CMD                  : 1;
  __REG16                       : 7;
} __ic_data_cmd_bits;

/* IC_INTR_STAT register */
typedef struct
{
  __REG16  R_RX_UNDER           : 1;
  __REG16  R_RX_OVER            : 1;
  __REG16  R_RX_FULL            : 1;
  __REG16  R_TX_OVER            : 1;
  __REG16  R_TX_EMPTY           : 1;
  __REG16  R_RD_REQ             : 1;
  __REG16  R_TX_ABRT            : 1;
  __REG16  R_RX_DONE            : 1;
  __REG16  R_ACTIVITY           : 1;
  __REG16  R_STOP_DET           : 1;
  __REG16  R_START_DET          : 1;
  __REG16  R_GEN_CALL           : 1;
  __REG16                       : 4;
} __ic_intr_stat_bits;

/* IC_INTR_MASK register */
typedef struct
{
  __REG16  M_RX_UNDER           : 1;
  __REG16  M_RX_OVER            : 1;
  __REG16  M_RX_FULL            : 1;
  __REG16  M_TX_OVER            : 1;
  __REG16  M_TX_EMPTY           : 1;
  __REG16  M_RD_REQ             : 1;
  __REG16  M_TX_ABRT            : 1;
  __REG16  M_RX_DONE            : 1;
  __REG16  M_ACTIVITY           : 1;
  __REG16  M_STOP_DET           : 1;
  __REG16  M_START_DET          : 1;
  __REG16  M_GEN_CALL           : 1;
  __REG16                       : 4;
} __ic_intr_mask_bits;

/* IC_RAW_INTR_STAT register */
typedef struct
{
  __REG16  RX_UNDER           : 1;
  __REG16  RX_OVER            : 1;
  __REG16  RX_FULL            : 1;
  __REG16  TX_OVER            : 1;
  __REG16  TX_EMPTY           : 1;
  __REG16  RD_REQ             : 1;
  __REG16  TX_ABRT            : 1;
  __REG16  RX_DONE            : 1;
  __REG16  ACTIVITY           : 1;
  __REG16  STOP_DET           : 1;
  __REG16  START_DET          : 1;
  __REG16  GEN_CALL           : 1;
  __REG16                     : 4;
} __ic_raw_intr_stat_bits;

/* IC_RX_TL register */
typedef struct
{
  __REG16  RX_TL              : 8;
  __REG16                     : 8;
} __ic_rx_tl_bits;

/* IC_TX_TL register */
typedef struct
{
  __REG16  TX_TL              : 8;
  __REG16                     : 8;
} __ic_tx_tl_bits;

/* IC_CLR_INTR register */
typedef struct
{
  __REG16  CLR_INTR           : 1;
  __REG16                     :15;
} __ic_clr_intr_bits;

/* IC_ENABLE register */
typedef struct
{
  __REG16  ENABLE             : 1;
  __REG16                     :15;
} __ic_enable_bits;

/* IC_STATUS register */
typedef struct
{
  __REG16  ACTIVITY           : 1;
  __REG16  TFNF               : 1;
  __REG16  TFE                : 1;
  __REG16  RFNE               : 1;
  __REG16  RFF                : 1;
  __REG16  MST_ACTIVITY       : 1;
  __REG16  SLV_ACTIVITY       : 1;
  __REG16                     : 9;
} __ic_status_bits;

/* IC_TXFLR register */
typedef struct
{
  __REG16  TXFLR              : 4;
  __REG16                     :12;
} __ic_txflr_bits;

/* IC_RXFLR register */
typedef struct
{
  __REG16  RXFLR              : 4;
  __REG16                     :12;
} __ic_rxflr_bits;

/* IC_TX_ABRT_SOURCE register */
typedef struct
{
  __REG16  ABRT_7B_ADDR_NOACK   : 1;
  __REG16  ABRT_10ADDR1_NOACK   : 1;
  __REG16  ABRT_10ADDR2_NOACK   : 1;
  __REG16  ABRT_TXDATA_NOACK    : 1;
  __REG16  ABRT_GCALL_NOACK     : 1;
  __REG16  ABRT_GCALL_READ      : 1;
  __REG16  ABRT_HS_ACKDET       : 1;
  __REG16  ABRT_SBYTE_ACKDET    : 1;
  __REG16  ABRT_HS_NORSTRT      : 1;
  __REG16  ABRT_SBYTE_NORSTRT   : 1;
  __REG16  ABRT_10B_RD_NORSTRT  : 1;
  __REG16  ARB_MASTER_DIS       : 1;
  __REG16  ARB_LOST             : 1;
  __REG16  ABRT_SLVFLUSH_TXFIFO : 1;
  __REG16  ABRT_SLV_ARBLOST     : 1;
  __REG16  ABRT_SLVRD_INTX      : 1;
} __ic_tx_abrt_source_bits;

/* IC_DMA_CR register */
typedef struct
{
  __REG16  RDMAE                : 1;
  __REG16  TDMAE                : 1;
  __REG16                       :14;
} __ic_dma_cr_bits;

/* IC_DMA_TDLR register */
typedef struct
{
  __REG16  DMATDL               : 3;
  __REG16                       :13;
} __ic_dma_tdlr_bits;

/* IC_DMA_RDLR register */
typedef struct
{
  __REG16  DMARDL               : 3;
  __REG16                       :13;
} __ic_dma_rdlr_bits;

/* IC_COMP_PARAM1 register */
typedef struct
{
  __REG32  APB_DATA_WIDTH       : 2;
  __REG32  MAX_SPEED_MODE       : 2;
  __REG32  HC_COUNT_VALUES      : 1;
  __REG32  INTR_IO              : 1;
  __REG32  HAS_DMA              : 1;
  __REG32  ADD_ENCODED_PARAMS   : 1;
  __REG32  RX_BUFFER_DEPTH      : 8;
  __REG32  TX_BUFFER_DEPTH      : 8;
  __REG32                       : 8;
} __ic_comp_param_1_bits;

/* ADC_STATUS_REG register */
typedef struct
{
  __REG16  E            : 1;
  __REG16  CS           : 3;
  __REG16  PD           : 1;
  __REG16  NOAS         : 3;
  __REG16  CR           : 1;
  __REG16  VRS          : 1;
  __REG16  EM           : 1;
  __REG16  ESR          : 1;
  __REG16  DE           : 1;
  __REG16  HR           : 1;
  __REG16               : 2;
} __adc_status_reg_bits;

/* ADC_AVERAGE_REG register */
typedef struct
{
  __REG16  CD           :10;
  __REG16               : 6;
} __adc_average_reg_bits;

/* ADC_CLK_REG register */
typedef struct
{
  __REG16  ADC_CLK_L    : 4;
  __REG16  ADC_CLK_H    : 4;
  __REG16               : 8;
} __adc_clk_reg_bits;

/* CHx CTRL register */
typedef struct
{
  __REG16  CE           : 1;
  __REG16  A            : 3;
  __REG16               :12;
} __adc_ch_ctrl_reg_bits;

/* CHx DATA register */
typedef struct
{
  __REG32  CD           :16;
  __REG32  DV           : 1;
  __REG32               :15;
} __adc_ch_data_reg_bits;

/* Boot register */
typedef struct
{
  __REG32  BD           : 3;
  __REG32  EMICS0BW     : 2;
  __REG32  NANDBW       : 1;
  __REG32  EMIACKB      : 1;
  __REG32               :25;
} __ras_br_bits;

/* Interrupt status register */
typedef struct
{
  __REG32  ISSMII1        : 1;
  __REG32  ISSMII2        : 1;
  __REG32  ISSMII3        : 1;
  __REG32  ISSMII4        : 1;
  __REG32  WUSMII1        : 1;
  __REG32  WUSMII2        : 1;
  __REG32  WUSMII3        : 1;
  __REG32  WUSMII4        : 1;
  __REG32  ISUART1        : 1;
  __REG32  ISUART2        : 1;
  __REG32  ISUART3        : 1;
  __REG32  ISUART4        : 1;
  __REG32  ISUART5        : 1;
  __REG32  EMI            : 1;
  __REG32  TDM            : 1;
  __REG32  RS485_1        : 1;
  __REG32  RS485_2        : 1;
  __REG32                 :15;
} __ras_isr_bits;

/* Function enable register */
typedef struct
{
  __REG32  TimeA        : 1;
  __REG32  TimeB        : 1;
  __REG32  UARTB        : 1;
  __REG32  UARTE        : 1;
  __REG32  GPIO5        : 1;
  __REG32  GPIO4        : 1;
  __REG32  GPIO3        : 1;
  __REG32  GPIO2        : 1;
  __REG32  GPIO1        : 1;
  __REG32  GPIO0        : 1;
  __REG32               : 1;
  __REG32  SSPB         : 1;
  __REG32  SSPE         : 1;
  __REG32  I2C          : 1;
  __REG32  FIrDA        : 1;
  __REG32               :17;
} __ras_fer_bits;

/* SMII MAC configuration register */
typedef struct
{
  __REG32  PHYM           : 2;
  __REG32  EndianessSMII1 : 1;
  __REG32  EndianessSMII2 : 1;
  __REG32  EndianessSMII3 : 1;
  __REG32  EndianessSMII4 : 1;
  __REG32                 :26;
} __ras_smiicfg_bits;

/* PL_GPIO_En0 */
typedef struct
{
  __REG32  PL_GPIO0       : 1;
  __REG32  PL_GPIO1       : 1;
  __REG32  PL_GPIO2       : 1;
  __REG32  PL_GPIO3       : 1;
  __REG32  PL_GPIO4       : 1;
  __REG32  PL_GPIO5       : 1;
  __REG32  PL_GPIO6       : 1;
  __REG32  PL_GPIO7       : 1;
  __REG32  PL_GPIO8       : 1;
  __REG32  PL_GPIO9       : 1;
  __REG32  PL_GPIO10      : 1;
  __REG32  PL_GPIO11      : 1;
  __REG32  PL_GPIO12      : 1;
  __REG32  PL_GPIO13      : 1;
  __REG32  PL_GPIO14      : 1;
  __REG32  PL_GPIO15      : 1;
  __REG32  PL_GPIO16      : 1;
  __REG32  PL_GPIO17      : 1;
  __REG32  PL_GPIO18      : 1;
  __REG32  PL_GPIO19      : 1;
  __REG32  PL_GPIO20      : 1;
  __REG32  PL_GPIO21      : 1;
  __REG32  PL_GPIO22      : 1;
  __REG32  PL_GPIO23      : 1;
  __REG32  PL_GPIO24      : 1;
  __REG32  PL_GPIO25      : 1;
  __REG32  PL_GPIO26      : 1;
  __REG32  PL_GPIO27      : 1;
  __REG32  PL_GPIO28      : 1;
  __REG32  PL_GPIO29      : 1;
  __REG32  PL_GPIO30      : 1;
  __REG32  PL_GPIO31      : 1;
} __ras_pl_gpio_en0_bits;

/* PL_GPIO_En1 */
typedef struct
{
  __REG32  PL_GPIO32      : 1;
  __REG32  PL_GPIO33      : 1;
  __REG32  PL_GPIO34      : 1;
  __REG32  PL_GPIO35      : 1;
  __REG32  PL_GPIO36      : 1;
  __REG32  PL_GPIO37      : 1;
  __REG32  PL_GPIO38      : 1;
  __REG32  PL_GPIO39      : 1;
  __REG32  PL_GPIO40      : 1;
  __REG32  PL_GPIO41      : 1;
  __REG32  PL_GPIO42      : 1;
  __REG32  PL_GPIO43      : 1;
  __REG32  PL_GPIO44      : 1;
  __REG32  PL_GPIO45      : 1;
  __REG32  PL_GPIO46      : 1;
  __REG32  PL_GPIO47      : 1;
  __REG32  PL_GPIO48      : 1;
  __REG32  PL_GPIO49      : 1;
  __REG32  PL_GPIO50      : 1;
  __REG32  PL_GPIO51      : 1;
  __REG32  PL_GPIO52      : 1;
  __REG32  PL_GPIO53      : 1;
  __REG32  PL_GPIO54      : 1;
  __REG32  PL_GPIO55      : 1;
  __REG32  PL_GPIO56      : 1;
  __REG32  PL_GPIO57      : 1;
  __REG32  PL_GPIO58      : 1;
  __REG32  PL_GPIO59      : 1;
  __REG32  PL_GPIO60      : 1;
  __REG32  PL_GPIO61      : 1;
  __REG32  PL_GPIO62      : 1;
  __REG32  PL_GPIO63      : 1;
} __ras_pl_gpio_en1_bits;

/* PL_GPIO_En2 */
typedef struct
{
  __REG32  PL_GPIO64      : 1;
  __REG32  PL_GPIO65      : 1;
  __REG32  PL_GPIO66      : 1;
  __REG32  PL_GPIO67      : 1;
  __REG32  PL_GPIO68      : 1;
  __REG32  PL_GPIO69      : 1;
  __REG32  PL_GPIO70      : 1;
  __REG32  PL_GPIO71      : 1;
  __REG32  PL_GPIO72      : 1;
  __REG32  PL_GPIO73      : 1;
  __REG32  PL_GPIO74      : 1;
  __REG32  PL_GPIO75      : 1;
  __REG32  PL_GPIO76      : 1;
  __REG32  PL_GPIO77      : 1;
  __REG32  PL_GPIO78      : 1;
  __REG32  PL_GPIO79      : 1;
  __REG32  PL_GPIO80      : 1;
  __REG32  PL_GPIO81      : 1;
  __REG32  PL_GPIO82      : 1;
  __REG32  PL_GPIO83      : 1;
  __REG32  PL_GPIO84      : 1;
  __REG32  PL_GPIO85      : 1;
  __REG32  PL_GPIO86      : 1;
  __REG32  PL_GPIO87      : 1;
  __REG32  PL_GPIO88      : 1;
  __REG32  PL_GPIO89      : 1;
  __REG32  PL_GPIO90      : 1;
  __REG32  PL_GPIO91      : 1;
  __REG32  PL_GPIO92      : 1;
  __REG32  PL_GPIO93      : 1;
  __REG32  PL_GPIO94      : 1;
  __REG32  PL_GPIO95      : 1;
} __ras_pl_gpio_en2_bits;

/* PL_GPIO_En3 */
typedef struct
{
  __REG32  PL_GPIO96      : 1;
  __REG32  PL_GPIO97      : 1;
  __REG32  PLCLK1         : 1;
  __REG32  PLCLK2         : 1;
  __REG32  PLCLK3         : 1;
  __REG32  PLCLK4         : 1;
  __REG32                 :26;
} __ras_pl_gpio_en3_bits;

/* GPIO_DataOut0 */
/* GPIO_Dir0 */
/* GPIO_IntEn0 */
/* GPIO_DataIn0 */
/* GPIO_IntSts0 */
typedef struct
{
  __REG32  PLCLK4         : 1;
  __REG32  PLCLK3         : 1;
  __REG32  PLCLK2         : 1;
  __REG32  PLCLK1         : 1;
  __REG32  PL_GPIO0       : 1;
  __REG32  PL_GPIO1       : 1;
  __REG32  PL_GPIO2       : 1;
  __REG32  PL_GPIO3       : 1;
  __REG32  PL_GPIO4       : 1;
  __REG32  PL_GPIO5       : 1;
  __REG32  PL_GPIO6       : 1;
  __REG32  PL_GPIO7       : 1;
  __REG32  PL_GPIO8       : 1;
  __REG32  PL_GPIO9       : 1;
  __REG32  PL_GPIO10      : 1;
  __REG32  PL_GPIO11      : 1;
  __REG32  PL_GPIO12      : 1;
  __REG32  PL_GPIO13      : 1;
  __REG32  PL_GPIO14      : 1;
  __REG32  PL_GPIO15      : 1;
  __REG32  PL_GPIO16      : 1;
  __REG32  PL_GPIO17      : 1;
  __REG32  PL_GPIO18      : 1;
  __REG32  PL_GPIO19      : 1;
  __REG32  PL_GPIO20      : 1;
  __REG32  PL_GPIO21      : 1;
  __REG32  PL_GPIO22      : 1;
  __REG32  PL_GPIO23      : 1;
  __REG32  PL_GPIO24      : 1;
  __REG32  PL_GPIO25      : 1;
  __REG32  PL_GPIO26      : 1;
  __REG32  PL_GPIO27      : 1;
} __ras_gpio_dataout0_bits;

/* GPIO_DataOut1 */
/* GPIO_Dir1 */
/* GPIO_IntEn1 */
/* GPIO_DataIn1 */
/* GPIO_IntSts1 */
typedef struct
{
  __REG32  PL_GPIO34      : 1;
  __REG32  PL_GPIO35      : 1;
  __REG32  PL_GPIO36      : 1;
  __REG32  PL_GPIO37      : 1;
  __REG32  PL_GPIO38      : 1;
  __REG32  PL_GPIO39      : 1;
  __REG32  PL_GPIO40      : 1;
  __REG32  PL_GPIO41      : 1;
  __REG32  PL_GPIO42      : 1;
  __REG32  PL_GPIO43      : 1;
  __REG32  PL_GPIO44      : 1;
  __REG32  PL_GPIO45      : 1;
  __REG32  PL_GPIO46      : 1;
  __REG32  PL_GPIO47      : 1;
  __REG32  PL_GPIO48      : 1;
  __REG32  PL_GPIO49      : 1;
  __REG32  PL_GPIO50      : 1;
  __REG32  PL_GPIO51      : 1;
  __REG32  PL_GPIO52      : 1;
  __REG32  PL_GPIO53      : 1;
  __REG32  PL_GPIO54      : 1;
  __REG32  PL_GPIO55      : 1;
  __REG32  PL_GPIO56      : 1;
  __REG32  PL_GPIO57      : 1;
  __REG32  PL_GPIO58      : 1;
  __REG32  PL_GPIO59      : 1;
  __REG32  PL_GPIO60      : 1;
  __REG32  PL_GPIO61      : 1;
  __REG32  PL_GPIO62      : 1;
  __REG32  PL_GPIO63      : 1;
  __REG32  PL_GPIO64      : 1;
  __REG32  PL_GPIO65      : 1;
} __ras_gpio_dataout1_bits;

/* GPIO_DataOut2 */
/* GPIO_Dir2 */
/* GPIO_IntEn2 */
/* GPIO_DataIn2 */
/* GPIO_IntSts2 */
typedef struct
{
  __REG32  PL_GPIO66      : 1;
  __REG32  PL_GPIO67      : 1;
  __REG32  PL_GPIO68      : 1;
  __REG32  PL_GPIO69      : 1;
  __REG32  PL_GPIO70      : 1;
  __REG32  PL_GPIO71      : 1;
  __REG32  PL_GPIO72      : 1;
  __REG32  PL_GPIO73      : 1;
  __REG32  PL_GPIO74      : 1;
  __REG32  PL_GPIO75      : 1;
  __REG32  PL_GPIO76      : 1;
  __REG32  PL_GPIO77      : 1;
  __REG32  PL_GPIO78      : 1;
  __REG32  PL_GPIO79      : 1;
  __REG32  PL_GPIO80      : 1;
  __REG32  PL_GPIO81      : 1;
  __REG32  PL_GPIO82      : 1;
  __REG32  PL_GPIO83      : 1;
  __REG32  PL_GPIO84      : 1;
  __REG32  PL_GPIO85      : 1;
  __REG32  PL_GPIO86      : 1;
  __REG32  PL_GPIO87      : 1;
  __REG32  PL_GPIO88      : 1;
  __REG32  PL_GPIO89      : 1;
  __REG32  PL_GPIO90      : 1;
  __REG32  PL_GPIO91      : 1;
  __REG32  PL_GPIO92      : 1;
  __REG32  PL_GPIO93      : 1;
  __REG32  PL_GPIO94      : 1;
  __REG32  PL_GPIO95      : 1;
  __REG32  PL_GPIO96      : 1;
  __REG32  PL_GPIO97      : 1;
} __ras_gpio_dataout2_bits;

/* GenMemCtrl_PC(i) registers */
typedef struct
{
  __REG32  Reset          : 1;
  __REG32  Wait_on        : 1;
  __REG32  Enable         : 1;
  __REG32  Dev_type       : 1;
  __REG32  Dev_width      : 2;
  __REG32  Eccen          : 1;
  __REG32  Eccplen        : 1;
  __REG32                 : 1;
  __REG32  tclr           : 4;
  __REG32  tar            : 4;
  __REG32                 :15;
} __genmemctrl_pc_bits;

/* GenMemCtrl_Comm0/GenMemCtrl_Attrib0/GenMemCtrl_I/O0 */
typedef struct
{
  __REG32  Tset           : 8;
  __REG32  Twait          : 8;
  __REG32  Thold          : 8;
  __REG32  Thiz           : 8;
} __genmemctrl_comm_bits;

/* GenMemCtrl_ECCr0 registers */
typedef struct
{
  __REG32  ECC1           : 8;
  __REG32  ECC2           : 8;
  __REG32  ECC3           : 8;
  __REG32                 : 8;
} __genmemctrl_eccr_bits;

/* Network control register */
typedef struct
{
  __REG32 LP             : 1;
  __REG32 LL             : 1;
  __REG32 RXE            : 1;
  __REG32 TXE            : 1;
  __REG32 MPE            : 1;
  __REG32 CS             : 1;
  __REG32 IS             : 1;
  __REG32 WEFSR          : 1;
  __REG32 BP             : 1;
  __REG32 ST             : 1;
  __REG32 TH             : 1;
  __REG32 TPF            : 1;
  __REG32 TZQPF          : 1;
  __REG32                :19;
} __macb_nctrl_bits;

/* Network configuration register */
typedef struct
{
  __REG32 SPEED          : 1;
  __REG32 FD             : 1;
  __REG32 BR             : 1;
  __REG32 JF             : 1;
  __REG32 CAF            : 1;
  __REG32 NB             : 1;
  __REG32 MHE            : 1;
  __REG32 UHE            : 1;
  __REG32 NJF            : 1;
  __REG32 EAME           : 1;
  __REG32 MDCDIV         : 2;
  __REG32 RE             : 1;
  __REG32 PE             : 1;
  __REG32 RBO            : 2;
  __REG32 RLFCE          : 1;
  __REG32 DRFCS          : 1;
  __REG32 HDE            : 1;
  __REG32 IRXFCS         : 1;
  __REG32 RBP            : 1;
  __REG32 DPFC           : 1;
  __REG32 MDCDIVH        : 1;
  __REG32                : 9;
} __macb_ncfg_bits;

/* Network status register */
typedef struct
{
  __REG32 LINK           : 1;
  __REG32 MDIO           : 1;
  __REG32 PHY_IDLE       : 1;
  __REG32                :29;
} __macb_nst_bits;

/* Revision register */
typedef struct
{
  __REG32 REV_REF        :16;
  __REG32 PART_REF       :16;
} __macb_rr_bits;

/* Transmit status register */
typedef struct
{
  __REG32 USED           : 1;
  __REG32 CO             : 1;
  __REG32 RLE            : 1;
  __REG32 GO             : 1;
  __REG32 BEMF           : 1;
  __REG32 TC             : 1;
  __REG32 TU             : 1;
  __REG32                :25;
} __macb_tst_bits;

/* Receive status register */
typedef struct
{
  __REG32 BNA            : 1;
  __REG32 FR             : 1;
  __REG32 RO             : 1;
  __REG32                :29;
} __macb_rst_bits;

/* Receive status register */
typedef struct
{
  __REG32 MFS            : 1;
  __REG32 RC             : 1;
  __REG32 RXU            : 1;
  __REG32 TXU            : 1;
  __REG32 TBU            : 1;
  __REG32 RLE            : 1;
  __REG32 TBEMF          : 1;
  __REG32 TC             : 1;
  __REG32                : 1;
  __REG32 LC             : 1;
  __REG32 RO             : 1;
  __REG32 HRESPNOK       : 1;
  __REG32 PFR            : 1;
  __REG32 PTZ            : 1;
  __REG32                :18;
} __macb_is_bits;

/* PHY maintenance register */
typedef struct
{
  __REG32 DATA           :16;
  __REG32 MBW2           : 2;
  __REG32 REGADDR        : 5;
  __REG32 PHYADDR        : 5;
  __REG32 CMD            : 2;
  __REG32 SOF            : 2;
} __macb_phym_bits;

/* Pause time register */
typedef struct
{
  __REG32 PT             :16;
  __REG32                :16;
} __macb_pt_bits;

/* Transmit pause quantum */
typedef struct
{
  __REG32 TPQ            :16;
  __REG32                :16;
} __macb_tpq_bits;

/* Wake-on LAN register */
typedef struct
{
  __REG32 IP             :16;
  __REG32 WOLEMP         : 1;
  __REG32 WOLEARP        : 1;
  __REG32 WOLSAEE        : 1;
  __REG32 WOLMHEE        : 1;
  __REG32                :12;
} __macb_wolan_bits;

/* Specific address 1 top */
typedef struct
{
  __REG32 ADDR           :16;
  __REG32                :16;
} __macb_sat1_bits;

/* Specific address 2 top */
typedef struct
{
  __REG32 ADDR           :16;
  __REG32                :16;
} __macb_sat2_bits;

/* Specific address 3 top */
typedef struct
{
  __REG32 ADDR           :16;
  __REG32                :16;
} __macb_sat3_bits;

/* Specific address 4 top */
typedef struct
{
  __REG32 ADDR           :16;
  __REG32                :16;
} __macb_sat4_bits;

/* Type ID checking */
typedef struct
{
  __REG32 ID             :16;
  __REG32                :16;
} __macb_tidc_bits;

/* Pause frames received */
typedef struct
{
  __REG32 FRAME_CNTR     :16;
  __REG32                :16;
} __macb_pfr_bits;

/* Frames received OK */
typedef struct
{
  __REG32 FRAME_CNTR     :24;
  __REG32                : 8;
} __macb_fr_bits;

/* Frames check sequence errors */
typedef struct
{
  __REG32 FCSE           : 8;
  __REG32                :24;
} __macb_fcse_bits;

/* Alignment errors */
typedef struct
{
  __REG32 AE             : 8;
  __REG32                :24;
} __macb_ae_bits;

/* Deferred transmission frames */
typedef struct
{
  __REG32 DTF            :16;
  __REG32                :16;
} __macb_dtf_bits;

/* Late collisions */
typedef struct
{
  __REG32 LC             : 8;
  __REG32                :24;
} __macb_lc_bits;

/* Excessive collisions */
typedef struct
{
  __REG32 EC             : 8;
  __REG32                :24;
} __macb_ec_bits;

/* Transmit underrun errors */
typedef struct
{
  __REG32 TUE            : 8;
  __REG32                :24;
} __macb_tue_bits;

/* Carrier sense errors */
typedef struct
{
  __REG32 CSE            : 8;
  __REG32                :24;
} __macb_cse_bits;

/* Receive resource errors */
typedef struct
{
  __REG32 RRE            :16;
  __REG32                :16;
} __macb_rre_bits;

/* Receive overrun errors */
typedef struct
{
  __REG32 ROE            : 8;
  __REG32                :24;
} __macb_roe_bits;

/* Receive symbol errors */
typedef struct
{
  __REG32 REE            : 8;
  __REG32                :24;
} __macb_rse_bits;

/* Excessive length errors */
typedef struct
{
  __REG32 ELE            : 8;
  __REG32                :24;
} __macb_ele_bits;

/* Receive Jabbers */
typedef struct
{
  __REG32 RJ             : 8;
  __REG32                :24;
} __macb_rj_bits;

/* Undersize frames */
typedef struct
{
  __REG32 UF             : 8;
  __REG32                :24;
} __macb_uf_bits;

/* SQE test errors */
typedef struct
{
  __REG32 SQETE          : 8;
  __REG32                :24;
} __macb_seqte_bits;

/* Received length field mismatch */
typedef struct
{
  __REG32 RLFM           : 8;
  __REG32                :24;
} __macb_rlfm_bits;

/* Transmitted pause frames */
typedef struct
{
  __REG32 TPF            :16;
  __REG32                :16;
} __macb_tpf_bits;

/* HDLC transmit command register */
typedef struct
{
  __REG32 C              : 2;
  __REG32 P              : 2;
  __REG32 F              : 1;
  __REG32 NCRC           : 1;
  __REG32 CSMA           : 1;
  __REG32 PEN            : 1;
  __REG32 CF             : 1;
  __REG32                : 1;
  __REG32 READ           : 1;
  __REG32 FALL_EDGE      : 1;
  __REG32                : 4;
  __REG32 CTS_DELAY      : 9;
  __REG32                : 7;
} __hdlc_htcr_bits;

/* HDLC receive command register */
typedef struct
{
  __REG32 C              : 2;
  __REG32 P              : 2;
  __REG32 F              : 1;
  __REG32 CRC            : 1;
  __REG32 AR10           : 1;
  __REG32 AR11           : 1;
  __REG32 AR20           : 1;
  __REG32 AR21           : 1;
  __REG32 READ           : 1;
  __REG32 RISE_EDGE      : 1;
  __REG32                :20;
} __hdlc_hrcr_bits;

/* Address field recognition address register */
typedef struct
{
  __REG32                : 9;
  __REG32 AMM            : 1;
  __REG32 READ           : 1;
  __REG32                :21;
} __hdlc_afrar_bits;

/* Interrupt mask register */
typedef struct
{
  __REG32 HDLC_MASK      : 1;
  __REG32 COV_MASK       : 1;
  __REG32 RxFWAR_MASK    : 1;
  __REG32 RxFOV_MASK     : 1;
  __REG32 TxFWAR_MASK    : 1;
  __REG32 TxFOV_MASK     : 1;
  __REG32 INTFWAR_MASK   : 1;
  __REG32 INTFOV_MASK    : 1;
  __REG32                :24;
} __hdlc_imr_bits;

/* Interrupt register */
typedef struct
{
  __REG32 HDLC           : 1;
  __REG32 ICOV           : 1;
  __REG32 RxFWAR         : 1;
  __REG32 RxFOV          : 1;
  __REG32 TxFWAR         : 1;
  __REG32 TxFOV          : 1;
  __REG32 INTFWAR        : 1;
  __REG32 INTFOV         : 1;
  __REG32                :24;
} __hdlc_ir_bits;

/* Interrupt queue size register */
typedef struct
{
  __REG32 HS             : 2;
  __REG32                :30;
} __hdlc_iqsr_bits;

/* TX/RX timeslot assigner address register */
typedef struct
{
  __REG32 TIME_SLOTS     : 7;
  __REG32                : 1;
  __REG32 HDI            : 1;
  __REG32 BUSY           : 1;
  __REG32 READ           : 1;
  __REG32                : 1;
  __REG32 DELAY          : 2;
  __REG32                :18;
} __tdm_rx_taar_bits;

/* TX/RX Timeslot assigner data register */
typedef struct
{
  __REG32 CH             : 7;
  __REG32 VAL            : 8;
  __REG32 VAL_CB         : 1;
  __REG32                :16;
} __tdm_rx_tadr_bits;

/* HDLC transmit command register */
typedef struct
{
  __REG32 C              : 2;
  __REG32 P              : 1;
  __REG32 F              : 1;
  __REG32 CRC32          : 1;
  __REG32 NCRC           : 1;
  __REG32 CSMA           : 1;
  __REG32 PEN            : 1;
  __REG32 CF             : 1;
  __REG32                : 1;
  __REG32 READ           : 1;
  __REG32 BUSY           : 1;
  __REG32 CH             : 7;
  __REG32 CONF_EN        : 1;
  __REG32 FALL_EDGE      : 1;
  __REG32                :11;
} __tdm_htcr_bits;

/* HDLC receive command register */
typedef struct
{
  __REG32 C              : 2;
  __REG32 P              : 1;
  __REG32 FM             : 1;
  __REG32 CRC32          : 1;
  __REG32 CRC            : 1;
  __REG32 AR1            : 2;
  __REG32 AR2            : 2;
  __REG32 READ           : 1;
  __REG32 BUSY           : 1;
  __REG32 CH             : 7;
  __REG32 CONF_EN        : 1;
  __REG32 RISE_EDGE      : 1;
  __REG32                :11;
} __tdm_hrcr_bits;

/* Address field recognition address register */
typedef struct
{
  __REG32 CH             : 7;
  __REG32                : 1;
  __REG32 BUSY           : 1;
  __REG32 AMM            : 1;
  __REG32 READ           : 1;
  __REG32                :21;
} __tdm_afrar_bits;

/* Interrupt mask register */
typedef struct
{
  __REG32 HDLC_MASK      : 1;
  __REG32 ICOV_MASK      : 1;
  __REG32 RxFWAR_MASK    : 1;
  __REG32 RxFOV_MASK     : 1;
  __REG32 TxFWAR_MASK    : 1;
  __REG32 TxFOV_MASK     : 1;
  __REG32 INTFWAR_MASK   : 1;
  __REG32 INTFOV_MASK    : 1;
  __REG32                :24;
} __tdm_imr_bits;

/* Interrupt register */
typedef struct
{
  __REG32 HDLC           : 1;
  __REG32 ICOV           : 1;
  __REG32 RxFWAR         : 1;
  __REG32 RxFOV          : 1;
  __REG32 TxFWAR         : 1;
  __REG32 TxFOV          : 1;
  __REG32 INTFWAR        : 1;
  __REG32 INTFOV         : 1;
  __REG32                :24;
} __tdm_ir_bits;

/* Interrupt queue size register */
typedef struct
{
  __REG32 HS             : 2;
  __REG32                :30;
} __tdm_iqsr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */
/***************************************************************************/
/* Common declarations  ****************************************************/
/***************************************************************************
 **
 **  VIC
 **
 ***************************************************************************/
__IO_REG32_BIT(VICIRQSTATUS,      0xF1100000,__READ       ,__vicirqstatus_bits);
__IO_REG32_BIT(VICFIQSTATUS,      0xF1100004,__READ       ,__vicfiqstatus_bits);
__IO_REG32_BIT(VICRAWINTR,        0xF1100008,__READ       ,__vicrawintr_bits);
__IO_REG32_BIT(VICINTSELECT,      0xF110000C,__READ_WRITE ,__vicintselect_bits);
__IO_REG32_BIT(VICINTENABLE,      0xF1100010,__READ_WRITE ,__vicintenable_bits);
__IO_REG32_BIT(VICINTENCLEAR,     0xF1100014,__WRITE      ,__vicintenclear_bits);
__IO_REG32_BIT(VICSOFTINT,        0xF1100018,__READ_WRITE ,__vicsoftint_bits);
__IO_REG32_BIT(VICSOFTINTCLEAR,   0xF110001C,__WRITE      ,__vicsoftintclear_bits);
__IO_REG32_BIT(VICPROTECTION,     0xF1100020,__READ_WRITE ,__vicprotection_bits);
__IO_REG32(    VICVECTADDR,       0xF1100030,__READ_WRITE );
__IO_REG32(    VICDEFVECTADDR,    0xF1100034,__READ_WRITE );
__IO_REG32(    VICVECTADDR0,      0xF1100100,__READ_WRITE );
__IO_REG32(    VICVECTADDR1,      0xF1100104,__READ_WRITE );
__IO_REG32(    VICVECTADDR2,      0xF1100108,__READ_WRITE );
__IO_REG32(    VICVECTADDR3,      0xF110010C,__READ_WRITE );
__IO_REG32(    VICVECTADDR4,      0xF1100110,__READ_WRITE );
__IO_REG32(    VICVECTADDR5,      0xF1100114,__READ_WRITE );
__IO_REG32(    VICVECTADDR6,      0xF1100118,__READ_WRITE );
__IO_REG32(    VICVECTADDR7,      0xF110011C,__READ_WRITE );
__IO_REG32(    VICVECTADDR8,      0xF1100120,__READ_WRITE );
__IO_REG32(    VICVECTADDR9,      0xF1100124,__READ_WRITE );
__IO_REG32(    VICVECTADDR10,     0xF1100128,__READ_WRITE );
__IO_REG32(    VICVECTADDR11,     0xF110012C,__READ_WRITE );
__IO_REG32(    VICVECTADDR12,     0xF1100130,__READ_WRITE );
__IO_REG32(    VICVECTADDR13,     0xF1100134,__READ_WRITE );
__IO_REG32(    VICVECTADDR14,     0xF1100138,__READ_WRITE );
__IO_REG32(    VICVECTADDR15,     0xF110013C,__READ_WRITE );
__IO_REG32_BIT(VICVECTCNTL0,      0xF1100200,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL1,      0xF1100204,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL2,      0xF1100208,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL3,      0xF110020C,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL4,      0xF1100210,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL5,      0xF1100214,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL6,      0xF1100218,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL7,      0xF110021C,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL8,      0xF1100220,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL9,      0xF1100224,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL10,     0xF1100228,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL11,     0xF110022C,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL12,     0xF1100230,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL13,     0xF1100234,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL14,     0xF1100238,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VICVECTCNTL15,     0xF110023C,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG8(     VICPERIPHID0,      0xF1100FE0,__READ       );
__IO_REG8_BIT( VICPERIPHID1,      0xF1100FE4,__READ       ,__vicperiphid1_bits);
__IO_REG8_BIT( VICPERIPHID2,      0xF1100FE8,__READ       ,__vicperiphid2_bits);
__IO_REG8(     VICPERIPHID3,      0xF1100FEC,__READ       );
__IO_REG8(     VICPCELLID0,       0xF1100FF0,__READ       );
__IO_REG8(     VICPCELLID1,       0xF1100FF4,__READ       );
__IO_REG8(     VICPCELLID2,       0xF1100FF8,__READ       );
__IO_REG8(     VICPCELLID3,       0xF1100FFC,__READ       );

/***************************************************************************
 **
 **  MPMC
 **
 ***************************************************************************/
__IO_REG32_BIT(MEM0_CTL,          0xFC600000,__READ_WRITE ,__mem0_ctl_bits);
#define MEMCTL_AHB_SET_00       MEM0_CTL
#define MEMCTL_AHB_SET_00_bit   MEM0_CTL_bit
__IO_REG32_BIT(MEM1_CTL,          0xFC600004,__READ_WRITE ,__mem1_ctl_bits);
#define MEMCTL_AHB_SET_01       MEM1_CTL
#define MEMCTL_AHB_SET_01_bit   MEM1_CTL_bit
__IO_REG32_BIT(MEM2_CTL,          0xFC600008,__READ_WRITE ,__mem2_ctl_bits);
#define MEMCTL_RFSH_SET_00      MEM2_CTL
#define MEMCTL_RFSH_SET_00_bit  MEM2_CTL_bit
__IO_REG32_BIT(MEM3_CTL,          0xFC60000C,__READ_WRITE ,__mem3_ctl_bits);
#define MEMCTL_DLL_SET_00       MEM3_CTL
#define MEMCTL_DLL_SET_00_bit   MEM3_CTL_bit
__IO_REG32_BIT(MEM4_CTL,          0xFC600010,__READ_WRITE ,__mem4_ctl_bits);
#define MEMCTL_GP_00            MEM4_CTL
#define MEMCTL_GP_00_bit        MEM4_CTL_bit
__IO_REG32_BIT(MEM5_CTL,          0xFC600014,__READ_WRITE ,__mem5_ctl_bits);
#define MEMCTL_GP_01            MEM5_CTL
#define MEMCTL_GP_01_bit        MEM5_CTL_bit
__IO_REG32_BIT(MEM6_CTL,          0xFC600018,__READ_WRITE ,__mem6_ctl_bits);
#define MEMCTL_GP_02            MEM6_CTL
#define MEMCTL_GP_02_bit        MEM6_CTL_bit
__IO_REG32_BIT(MEM7_CTL,          0xFC60001C,__READ_WRITE ,__mem7_ctl_bits);
#define MEMCTL_GP_03            MEM7_CTL
#define MEMCTL_GP_03_bit        MEM7_CTL_bit
__IO_REG32_BIT(MEM8_CTL,          0xFC600020,__READ_WRITE ,__mem8_ctl_bits);
#define MEMCTL_GP_04            MEM8_CTL
#define MEMCTL_GP_04_bit        MEM8_CTL_bit
__IO_REG32_BIT(MEM9_CTL,          0xFC600024,__READ_WRITE ,__mem9_ctl_bits);
#define MEMCTL_GP_05            MEM9_CTL
#define MEMCTL_GP_05_bit        MEM9_CTL_bit
__IO_REG32_BIT(MEM10_CTL,         0xFC600028,__READ_WRITE ,__mem10_ctl_bits);
#define MEMCTL_GP_06            MEM10_CTL
#define MEMCTL_GP_06_bit        MEM10_CTL_bit
__IO_REG32_BIT(MEM11_CTL,         0xFC60002C,__READ_WRITE ,__mem11_ctl_bits);
#define MEMCTL_AHB_SET_02       MEM11_CTL
#define MEMCTL_AHB_SET_02_bit   MEM11_CTL_bit
__IO_REG32_BIT(MEM12_CTL,         0xFC600030,__READ_WRITE ,__mem12_ctl_bits);
#define MEMCTL_AHB_SET_03       MEM12_CTL
#define MEMCTL_AHB_SET_03_bit   MEM12_CTL_bit
__IO_REG32_BIT(MEM13_CTL,         0xFC600034,__READ_WRITE ,__mem13_ctl_bits);
#define MEMCTL_AHB_SET_04       MEM13_CTL
#define MEMCTL_AHB_SET_04_bit   MEM13_CTL_bit
__IO_REG32_BIT(MEM14_CTL,         0xFC600038,__READ_WRITE ,__mem14_ctl_bits);
#define MEMCTL_AHB_SET_05       MEM14_CTL
#define MEMCTL_AHB_SET_05_bit   MEM14_CTL_bit
__IO_REG32_BIT(MEM15_CTL,         0xFC60003C,__READ_WRITE ,__mem15_ctl_bits);
#define MEMCTL_AHB_SET_06       MEM15_CTL
#define MEMCTL_AHB_SET_06_bit   MEM15_CTL_bit
__IO_REG32_BIT(MEM17_CTL,         0xFC600044,__READ_WRITE ,__mem17_ctl_bits);
#define MEMCTL_DQS_SET_0        MEM17_CTL
#define MEMCTL_DQS_SET_0_bit    MEM17_CTL_bit
__IO_REG32_BIT(MEM18_CTL,         0xFC600048,__READ_WRITE ,__mem18_ctl_bits);
#define MEMCTL_TIME_SET_01      MEM18_CTL
#define MEMCTL_TIME_SET_01_bit  MEM18_CTL_bit
__IO_REG32_BIT(MEM19_CTL,         0xFC60004C,__READ_WRITE ,__mem19_ctl_bits);
#define MEMCTL_TIME_SET_02      MEM19_CTL
#define MEMCTL_TIME_SET_02_bit  MEM19_CTL_bit
__IO_REG32_BIT(MEM20_CTL,         0xFC600050,__READ_WRITE ,__mem20_ctl_bits);
#define MEMCTL_AHB_RELPR_00     MEM20_CTL
#define MEMCTL_AHB_RELPR_00_bit MEM20_CTL_bit
__IO_REG32_BIT(MEM21_CTL,         0xFC600054,__READ_WRITE ,__mem21_ctl_bits);
#define MEMCTL_AHB_RELPR_01     MEM21_CTL
#define MEMCTL_AHB_RELPR_01_bit MEM21_CTL_bit
__IO_REG32_BIT(MEM22_CTL,         0xFC600058,__READ_WRITE ,__mem22_ctl_bits);
#define MEMCTL_AHB_RELPR_02     MEM22_CTL
#define MEMCTL_AHB_RELPR_02_bit MEM22_CTL_bit
__IO_REG32_BIT(MEM23_CTL,         0xFC60005C,__READ_WRITE ,__mem23_ctl_bits);
#define MEMCTL_AHB_RELPR_03     MEM23_CTL
#define MEMCTL_AHB_RELPR_03_bit MEM23_CTL_bit
__IO_REG32_BIT(MEM24_CTL,         0xFC600060,__READ_WRITE ,__mem24_ctl_bits);
#define MEMCTL_AHB_RELPR_04     MEM24_CTL
#define MEMCTL_AHB_RELPR_04_bit MEM24_CTL_bit
__IO_REG32_BIT(MEM25_CTL,         0xFC600064,__READ_WRITE ,__mem25_ctl_bits);
#define MEMCTL_AHB_RELPR_05     MEM25_CTL
#define MEMCTL_AHB_RELPR_05_bit MEM25_CTL_bit
__IO_REG32_BIT(MEM26_CTL,         0xFC600068,__READ_WRITE ,__mem26_ctl_bits);
#define MEMCTL_AHB_RELPR_06     MEM26_CTL
#define MEMCTL_AHB_RELPR_06_bit MEM26_CTL_bit
__IO_REG32_BIT(MEM27_CTL,         0xFC60006C,__READ_WRITE ,__mem27_ctl_bits);
#define MEMCTL_AHB_RELPR_07     MEM27_CTL
#define MEMCTL_AHB_RELPR_07_bit MEM27_CTL_bit
__IO_REG32_BIT(MEM28_CTL,         0xFC600070,__READ_WRITE ,__mem28_ctl_bits);
#define MEMCTL_AHB_RELPR_08     MEM28_CTL
#define MEMCTL_AHB_RELPR_08_bit MEM28_CTL_bit
__IO_REG32_BIT(MEM29_CTL,         0xFC600074,__READ_WRITE ,__mem29_ctl_bits);
#define MEMCTL_AHB_RELPR_09     MEM29_CTL
#define MEMCTL_AHB_RELPR_09_bit MEM29_CTL_bit
__IO_REG32_BIT(MEM30_CTL,         0xFC600078,__READ_WRITE ,__mem30_ctl_bits);
#define MEMCTL_AHB_RELPR_10     MEM30_CTL
#define MEMCTL_AHB_RELPR_10_bit MEM30_CTL_bit
__IO_REG32_BIT(MEM34_CTL,         0xFC600088,__READ_WRITE ,__mem34_ctl_bits);
#define MEMCTL_DQS_SET_1        MEM34_CTL
#define MEMCTL_DQS_SET_1_bit    MEM34_CTL_bit
__IO_REG32_BIT(MEM35_CTL,         0xFC60008C,__READ_WRITE ,__mem35_ctl_bits);
#define MEMCTL_GP_07            MEM35_CTL
#define MEMCTL_GP_07_bit        MEM35_CTL_bit
__IO_REG32_BIT(MEM36_CTL,         0xFC600090,__READ_WRITE ,__mem36_ctl_bits);
#define MEMCTL_GP_08            MEM36_CTL
#define MEMCTL_GP_08_bit        MEM36_CTL_bit
__IO_REG32_BIT(MEM37_CTL,         0xFC600094,__READ_WRITE ,__mem37_ctl_bits);
#define MEMCTL_GP_09            MEM37_CTL
#define MEMCTL_GP_09_bit        MEM37_CTL_bit
__IO_REG32_BIT(MEM38_CTL,         0xFC600098,__READ_WRITE ,__mem38_ctl_bits);
#define MEMCTL_GP_10            MEM38_CTL
#define MEMCTL_GP_10_bit        MEM38_CTL_bit
__IO_REG32_BIT(MEM39_CTL,         0xFC60009C,__READ_WRITE ,__mem39_ctl_bits);
#define MEMCTL_DLL_SET_01       MEM39_CTL
#define MEMCTL_DLL_SET_01_bit   MEM39_CTL_bit
__IO_REG32_BIT(MEM40_CTL,         0xFC6000A0,__READ_WRITE ,__mem40_ctl_bits);
#define MEMCTL_DQS_OUT_SHIFT      MEM40_CTL
#define MEMCTL_DQS_OUT_SHIFT_bit  MEM40_CTL_bit
__IO_REG32_BIT(MEM41_CTL,         0xFC6000A4,__READ_WRITE ,__mem41_ctl_bits);
#define MEMCTL_WR_DQS_SHIFT     MEM41_CTL
#define MEMCTL_WR_DQS_SHIFT_bit MEM41_CTL_bit
__IO_REG32_BIT(MEM42_CTL,         0xFC6000A8,__READ_WRITE ,__mem42_ctl_bits);
#define MEMCTL_TIME_SET_03      MEM42_CTL
#define MEMCTL_TIME_SET_03_bit  MEM42_CTL_bit
__IO_REG32_BIT(MEM43_CTL,         0xFC6000AC,__READ_WRITE ,__mem43_ctl_bits);
#define MEMCTL_AHB_PRRLX_00     MEM43_CTL
#define MEMCTL_AHB_PRRLX_00_bit MEM43_CTL_bit
__IO_REG32_BIT(MEM44_CTL,         0xFC6000B0,__READ_WRITE ,__mem44_ctl_bits);
#define MEMCTL_AHB_PRRLX_01     MEM44_CTL
#define MEMCTL_AHB_PRRLX_01_bit MEM44_CTL_bit
__IO_REG32_BIT(MEM45_CTL,         0xFC6000B4,__READ_WRITE ,__mem45_ctl_bits);
#define MEMCTL_AHB_PRRLX_02     MEM45_CTL
#define MEMCTL_AHB_PRRLX_02_bit MEM45_CTL_bit
__IO_REG32_BIT(MEM46_CTL,         0xFC6000B8,__READ       ,__mem46_ctl_bits);
#define MEMCTL_OUTRANGE_LGTH      MEM46_CTL
#define MEMCTL_OUTRANGE_LGTH_bit  MEM46_CTL_bit
__IO_REG32_BIT(MEM47_CTL,         0xFC6000BC,__READ_WRITE ,__mem47_ctl_bits);
#define MEMCTL_AHB_RW_SET_00      MEM47_CTL
#define MEMCTL_AHB_RW_SET_00_bit  MEM47_CTL_bit
__IO_REG32_BIT(MEM48_CTL,         0xFC6000C0,__READ_WRITE ,__mem48_ctl_bits);
#define MEMCTL_AHB_RW_SET_01      MEM48_CTL
#define MEMCTL_AHB_RW_SET_01_bit  MEM48_CTL_bit
__IO_REG32_BIT(MEM49_CTL,         0xFC6000C4,__READ_WRITE ,__mem49_ctl_bits);
#define MEMCTL_AHB_RW_SET_02      MEM49_CTL
#define MEMCTL_AHB_RW_SET_02_bit  MEM49_CTL_bit
__IO_REG32_BIT(MEM50_CTL,         0xFC6000C8,__READ_WRITE ,__mem50_ctl_bits);
#define MEMCTL_AHB_RW_SET_03      MEM50_CTL
#define MEMCTL_AHB_RW_SET_03_bit  MEM50_CTL_bit
__IO_REG32_BIT(MEM51_CTL,         0xFC6000CC,__READ_WRITE ,__mem51_ctl_bits);
#define MEMCTL_AHB_RW_SET_04      MEM51_CTL
#define MEMCTL_AHB_RW_SET_04_bit  MEM51_CTL_bit
__IO_REG32_BIT(MEM54_CTL,         0xFC6000D8,__READ_WRITE ,__mem54_ctl_bits);
#define MEMCTL_TREF             MEM54_CTL
#define MEMCTL_TREF_bit         MEM54_CTL_bit
__IO_REG32_BIT(MEM55_CTL,         0xFC6000DC,__READ_WRITE ,__mem55_ctl_bits);
#define MEMCTL_EMRS3_DATA       MEM55_CTL
#define MEMCTL_EMRS3_DATA_bit   MEM55_CTL_bit
__IO_REG32_BIT(MEM56_CTL,         0xFC6000E0,__READ_WRITE ,__mem56_ctl_bits);
#define MEMCTL_TIME_SET_04      MEM56_CTL
#define MEMCTL_TIME_SET_04_bit  MEM56_CTL_bit
__IO_REG32_BIT(MEM57_CTL,         0xFC6000E4,__READ_WRITE ,__mem57_ctl_bits);
#define MEMCTL_TIME_SET_05      MEM57_CTL
#define MEMCTL_TIME_SET_05_bit  MEM57_CTL_bit
__IO_REG32_BIT(MEM58_CTL,         0xFC6000E8,__READ       ,__mem58_ctl_bits);
#define MEMCTL_VERSION          MEM58_CTL
#define MEMCTL_VERSION_bit      MEM58_CTL_bit
__IO_REG32_BIT(MEM59_CTL,         0xFC6000EC,__READ_WRITE ,__mem59_ctl_bits);
#define MEMCTL_TINIT            MEM59_CTL
#define MEMCTL_TINIT_bit        MEM59_CTL_bit
__IO_REG32(    MEM60_CTL,         0xFC6000F0,__READ       );
#define MEMCTL_OUTRANGE_ADDR_01     MEM60_CTL
__IO_REG32_BIT(MEM61_CTL,         0xFC6000F4,__READ       ,__mem61_ctl_bits);
#define MEMCTL_OUTRANGE_ADDR_02     MEM61_CTL
#define MEMCTL_OUTRANGE_ADDR_02_bit MEM61_CTL_bit
__IO_REG32_BIT(MEM65_CTL,         0xFC600104,__READ_WRITE ,__mem65_ctl_bits);
#define MEMCTL_DLL_DQS_DELAY_BYPASS_0     MEM65_CTL
#define MEMCTL_DLL_DQS_DELAY_BYPASS_0_bit MEM65_CTL_bit
__IO_REG32_BIT(MEM66_CTL,         0xFC600108,__READ_WRITE ,__mem66_ctl_bits);
#define MEMCTL_DLL_SET_02       MEM66_CTL
#define MEMCTL_DLL_SET_02_bit   MEM66_CTL_bit
__IO_REG32_BIT(MEM67_CTL,         0xFC60010C,__READ_WRITE ,__mem67_ctl_bits);
#define MEMCTL_DLL_SET_03       MEM67_CTL
#define MEMCTL_DLL_SET_03_bit   MEM67_CTL_bit
__IO_REG32_BIT(MEM68_CTL,         0xFC600110,__READ_WRITE ,__mem68_ctl_bits);
#define MEMCTL_DQS_SET_2        MEM68_CTL
#define MEMCTL_DQS_SET_2_bit    MEM68_CTL_bit
__IO_REG32(    MEM98_CTL,         0xFC600188,__READ_WRITE );
#define MEMCTL_USER_DEF_REG_0     MEM98_CTL
__IO_REG32(    MEM99_CTL,         0xFC60018C,__READ_WRITE );
#define MEMCTL_USER_DEF_REG_1     MEM99_CTL
__IO_REG32_BIT(MEM100_CTL,        0xFC600190,__READ_WRITE ,__mem100_ctl_bits);
#define MEMCTL_GP_11            MEM100_CTL
#define MEMCTL_GP_11_bit        MEM100_CTL_bit
__IO_REG32_BIT(MEM101_CTL,        0xFC600194,__READ_WRITE ,__mem101_ctl_bits);
#define MEMCTL_GP_12            MEM101_CTL
#define MEMCTL_GP_12_bit        MEM101_CTL_bit
__IO_REG32_BIT(MEM102_CTL,        0xFC600198,__READ_WRITE ,__mem102_ctl_bits);
#define MEMCTL_GP_13            MEM102_CTL
#define MEMCTL_GP_13_bit        MEM102_CTL_bit
__IO_REG32_BIT(MEM103_CTL,        0xFC60019C,__READ_WRITE ,__mem103_ctl_bits);
#define MEMCTL_GP_14            MEM103_CTL
#define MEMCTL_GP_14_bit        MEM103_CTL_bit
__IO_REG32_BIT(MEM104_CTL,        0xFC6001A0,__READ_WRITE ,__mem104_ctl_bits);
#define MEMCTL_EMRS2_DATA_X     MEM104_CTL
#define MEMCTL_EMRS2_DATA_X_bit MEM104_CTL_bit
__IO_REG32_BIT(MEM105_CTL,        0xFC6001A4,__READ_WRITE ,__mem105_ctl_bits);
#define MEMCTL_LWPWR_CNT        MEM105_CTL
#define MEMCTL_LWPWR_CNT_bit    MEM105_CTL_bit
__IO_REG32_BIT(MEM106_CTL,        0xFC6001A8,__READ_WRITE ,__mem106_ctl_bits);
#define MEMCTL_LWPWR_REG        MEM106_CTL
#define MEMCTL_LWPWR_REG_bit    MEM106_CTL_bit
__IO_REG32_BIT(MEM107_CTL,        0xFC6001AC,__READ_WRITE ,__mem107_ctl_bits);
#define MEMCTL_GP_15            MEM107_CTL
#define MEMCTL_GP_15_bit        MEM107_CTL_bit
__IO_REG32_BIT(MEM108_CTL,        0xFC6001B0,__READ_WRITE ,__mem108_ctl_bits);
#define MEMCTL_TPDEX            MEM108_CTL
#define MEMCTL_TPDEX_bit        MEM108_CTL_bit

/***************************************************************************
 **
 **  Misc
 **
 ***************************************************************************/
__IO_REG32_BIT(SOC_CFG_CTR,       0xFCA80000,__READ       ,__soc_cfg_ctr_bits);
__IO_REG32_BIT(DIAG_CFG_CTR,      0xFCA80004,__READ_WRITE ,__diag_cfg_ctr_bits);
__IO_REG32_BIT(PLL1_CTR,          0xFCA80008,__READ_WRITE ,__pll_ctr_bits);
__IO_REG32_BIT(PLL1_FRQ,          0xFCA8000C,__READ_WRITE ,__pll_frq_bits);
__IO_REG32_BIT(PLL1_MOD,          0xFCA80010,__READ_WRITE ,__pll_mod_bits);
__IO_REG32_BIT(PLL2_CTR,          0xFCA80014,__READ_WRITE ,__pll_ctr_bits);
__IO_REG32_BIT(PLL2_FRQ,          0xFCA80018,__READ_WRITE ,__pll_frq_bits);
__IO_REG32_BIT(PLL2_MOD,          0xFCA8001C,__READ_WRITE ,__pll_mod_bits);
__IO_REG32_BIT(PLL_CLK_CFG,       0xFCA80020,__READ_WRITE ,__pll_clk_cfg_bits);
__IO_REG32_BIT(CORE_CLK_CFG,      0xFCA80024,__READ_WRITE ,__core_clk_cfg_bits);
__IO_REG32_BIT(PRPH_CLK_CFG,      0xFCA80028,__READ_WRITE ,__prph_clk_cfg_bits);
__IO_REG32_BIT(PERIP1_CLK_ENB,    0xFCA8002C,__READ_WRITE ,__perip1_clk_enb_bits);
__IO_REG32_BIT(RAS_CLK_ENB,       0xFCA80034,__READ_WRITE ,__ras_clk_enb_bits);
__IO_REG32_BIT(PERIP1_SOF_RST,    0xFCA80038,__READ_WRITE ,__perip1_sof_rst_bits);
__IO_REG32_BIT(RAS_SOF_RST,       0xFCA80040,__READ_WRITE ,__ras_sof_rst_bits);
__IO_REG32_BIT(PRSC1_CLK_CFG,     0xFCA80044,__READ_WRITE ,__prsc_clk_cfg_bits);
__IO_REG32_BIT(PRSC2_CLK_CFG,     0xFCA80048,__READ_WRITE ,__prsc_clk_cfg_bits);
__IO_REG32_BIT(PRSC3_CLK_CFG,     0xFCA8004C,__READ_WRITE ,__prsc_clk_cfg_bits);
__IO_REG32_BIT(AMEM_CFG_CTRL,     0xFCA80050,__READ_WRITE ,__amem_cfg_ctrl_bits);
__IO_REG32_BIT(IRDA_CLK_SYNT,     0xFCA80060,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(UART_CLK_SYNT,     0xFCA80064,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(MAC_CLK_SYNT,      0xFCA80068,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(RAS1_CLK_SYNT,     0xFCA8006C,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(RAS2_CLK_SYNT,     0xFCA80070,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(RAS3_CLK_SYNT,     0xFCA80074,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(RAS4_CLK_SYNT,     0xFCA80078,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(ICM1_ARB_CFG,      0xFCA8007C,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM2_ARB_CFG,      0xFCA80080,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM3_ARB_CFG,      0xFCA80084,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM4_ARB_CFG,      0xFCA80088,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM5_ARB_CFG,      0xFCA8008C,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM6_ARB_CFG,      0xFCA80090,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM7_ARB_CFG,      0xFCA80094,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM8_ARB_CFG,      0xFCA80098,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM9_ARB_CFG,      0xFCA8009C,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(DMA_CHN_CFG,       0xFCA800A0,__READ_WRITE ,__dma_chn_cfg_bits);
__IO_REG32_BIT(USB2_PHY_CFG,      0xFCA800A4,__READ_WRITE ,__usb2_phy_cfg_bits);
__IO_REG32_BIT(MAC_CFG_CTR,       0xFCA800A8,__READ_WRITE ,__mac_cfg_ctr_bits);
__IO_REG32_BIT(PRC1_LOCK_CTR,     0xFCA800C0,__READ_WRITE ,__prc_lock_ctr_bits);
__IO_REG32_BIT(PRC2_LOCK_CTR,     0xFCA800C4,__READ_WRITE ,__prc_lock_ctr_bits);
__IO_REG32_BIT(PRC1_IRQ_CTR,      0xFCA800D0,__READ_WRITE ,__prc_irq_ctr_bits);
__IO_REG32_BIT(PRC2_IRQ_CTR,      0xFCA800D4,__READ_WRITE ,__prc_irq_ctr_bits);
__IO_REG32_BIT(POWERDOWN_CFG_CTR, 0xFCA800E0,__READ_WRITE ,__powerdown_cfg_ctr_bits);
__IO_REG32_BIT(COMPSSTL_1V8_CFG,  0xFCA800E4,__READ_WRITE ,__compsstl_1v8_cfg_bits);
__IO_REG32_BIT(COMPCOR_3V3_CFG,   0xFCA800EC,__READ_WRITE ,__compsstl_1v8_cfg_bits);
__IO_REG32_BIT(DDR_PAD,           0xFCA800F0,__READ_WRITE ,__ddr_pad_bits);
__IO_REG32_BIT(BIST1_CFG_CTR,     0xFCA800F4,__READ_WRITE ,__bist1_cfg_ctr_bits);
__IO_REG32_BIT(BIST2_CFG_CTR,     0xFCA800F8,__READ_WRITE ,__bist2_cfg_ctr_bits);
__IO_REG32_BIT(BIST3_CFG_CTR,     0xFCA800FC,__READ_WRITE ,__bist3_cfg_ctr_bits);
__IO_REG32_BIT(BIST4_CFG_CTR,     0xFCA80100,__READ_WRITE ,__bist4_cfg_ctr_bits);
__IO_REG32_BIT(BIST1_STS_RES,     0xFCA80108,__READ       ,__bist1_sts_res_bits);
__IO_REG32_BIT(BIST2_STS_RES,     0xFCA8010C,__READ       ,__bist2_sts_res_bits);
__IO_REG32_BIT(BIST3_STS_RES,     0xFCA80110,__READ       ,__bist3_sts_res_bits);
__IO_REG32_BIT(BIST4_STS_RES,     0xFCA80114,__READ       ,__bist4_sts_res_bits);
__IO_REG32_BIT(SYSERR_CFG_CTR,    0xFCA8011C,__READ_WRITE ,__syserr_cfg_ctr_bits);
__IO_REG32_BIT(USB0_TUN_PRM,      0xFCA80120,__READ_WRITE ,__usb_tun_prm_bits);
__IO_REG32_BIT(USB_TUN_PRM,       0xFCA80124,__READ_WRITE ,__usb_tun_prm_bits);
__IO_REG32_BIT(USB2_TUN_PRM,      0xFCA80128,__READ_WRITE ,__usb_tun_prm_bits);
__IO_REG32_BIT(PLGPIO0_PAD_PRG,   0xFCA80130,__READ_WRITE ,__plgpio0_pad_prg_bits);
__IO_REG32_BIT(PLGPIO1_PAD_PRG,   0xFCA80134,__READ_WRITE ,__plgpio1_pad_prg_bits);
__IO_REG32_BIT(PLGPIO2_PAD_PRG,   0xFCA80138,__READ_WRITE ,__plgpio2_pad_prg_bits);
__IO_REG32_BIT(PLGPIO3_PAD_PRG,   0xFCA8013C,__READ_WRITE ,__plgpio3_pad_prg_bits);
__IO_REG32_BIT(PLGPIO4_PAD_PRG,   0xFCA80140,__READ_WRITE ,__plgpio4_pad_prg_bits);

/***************************************************************************
 **
 **  SSP
 **
 ***************************************************************************/
__IO_REG16_BIT(SSPCR0,            0xD0100000,__READ_WRITE ,__sspcr0_bits);
__IO_REG16_BIT(SSPCR1,            0xD0100004,__READ_WRITE ,__sspcr1_bits);
__IO_REG16(    SSPDR,             0xD0100008,__READ_WRITE );
__IO_REG16_BIT(SSPSR,             0xD010000C,__READ       ,__sspsr_bits);
__IO_REG16_BIT(SSPCPSR,           0xD0100010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG16_BIT(SSPIMSC,           0xD0100014,__READ_WRITE ,__sspimsc_bits);
__IO_REG16_BIT(SSPRIS,            0xD0100018,__READ       ,__sspris_bits);
__IO_REG16_BIT(SSPMIS,            0xD010001C,__READ       ,__sspmis_bits);
__IO_REG16_BIT(SSPICR,            0xD0100020,__WRITE      ,__sspicr_bits);
__IO_REG16_BIT(SSPDMACR,          0xD0100024,__READ_WRITE ,__sspdmacr_bits);
__IO_REG16_BIT(SSPPeriphID0,      0xD0100FE0,__READ       ,__sspperiphid0_bits);
__IO_REG16_BIT(SSPPeriphID1,      0xD0100FE4,__READ       ,__sspperiphid1_bits);
__IO_REG16_BIT(SSPPeriphID2,      0xD0100FE8,__READ       ,__sspperiphid2_bits);
__IO_REG16_BIT(SSPPeriphID3,      0xD0100FEC,__READ       ,__sspperiphid3_bits);
__IO_REG16_BIT(SSPCellID0,        0xD0100FF0,__READ       ,__sspcellid0_bits);
__IO_REG16_BIT(SSPCellID1,        0xD0100FF4,__READ       ,__sspcellid1_bits);
__IO_REG16_BIT(SSPCellID2,        0xD0100FF8,__READ       ,__sspcellid2_bits);
__IO_REG16_BIT(SSPCellID3,        0xD0100FFC,__READ       ,__sspcellid3_bits);

/***************************************************************************
 **
 **  SC
 **
 ***************************************************************************/
__IO_REG32_BIT(SCCTRL,            0xFCA00000,__READ_WRITE ,__scctrl_bits);
__IO_REG32(    SCSYSSTAT,         0xFCA00004,__WRITE      );
__IO_REG32_BIT(SCIMCTRL,          0xFCA00008,__READ_WRITE ,__scimctrl_bits);
__IO_REG32_BIT(SCIMSTAT,          0xFCA0000C,__READ_WRITE ,__scimstat_bits);
__IO_REG32_BIT(SCXTALCTRL,        0xFCA00010,__READ_WRITE ,__scxtalctrl_bits);
__IO_REG32_BIT(SCPLLCTRL,         0xFCA00014,__READ_WRITE ,__scpllctrl_bits);
__IO_REG32(    SCSYSID0,          0xFCA00EE0,__READ       );
__IO_REG32(    SCSYSID1,          0xFCA00EE4,__READ       );
__IO_REG32(    SCSYSID2,          0xFCA00EE8,__READ       );
__IO_REG32(    SCSYSID3,          0xFCA00EEC,__READ       );
__IO_REG32(    SCPeriphID0,       0xFCA00FE0,__READ       );
__IO_REG32(    SCPeriphID1,       0xFCA00FE4,__READ       );
__IO_REG32(    SCPeriphID2,       0xFCA00FE8,__READ       );
__IO_REG32(    SCPeriphID3,       0xFCA00FEC,__READ       );
__IO_REG32(    SCPCellID0,        0xFCA00FF0,__READ       );
__IO_REG32(    SCPCellID1,        0xFCA00FF4,__READ       );
__IO_REG32(    SCPCellID2,        0xFCA00FF8,__READ       );
__IO_REG32(    SCPCellID3,        0xFCA00FFC,__READ       );

/***************************************************************************
 **
 **  SMI
 **
 ***************************************************************************/
__IO_REG32_BIT(SMI_CR1,           0xFC000000,__READ_WRITE ,__smi_cr1_bits);
__IO_REG32_BIT(SMI_CR2,           0xFC000004,__READ_WRITE ,__smi_cr2_bits);
__IO_REG32_BIT(SMI_SR,            0xFC000008,__READ_WRITE ,__smi_sr_bits);
__IO_REG32(    SMI_TR,            0xFC00000C,__READ_WRITE );
__IO_REG32(    SMI_RR,            0xFC000010,__READ_WRITE );

/***************************************************************************
 **
 **  WDT
 **
 ***************************************************************************/
__IO_REG32(    WdogLoad,          0xFC880000,__READ_WRITE );
__IO_REG32(    WdogValue,         0xFC880004,__READ       );
__IO_REG32_BIT(WdogControl,       0xFC880008,__READ_WRITE ,__wdogcontrol_bits);
__IO_REG32(    WdogIntClr,        0xFC88000C,__WRITE      );
__IO_REG32_BIT(WdogRIS,           0xFC880010,__READ       ,__wdogris_bits);
__IO_REG32_BIT(WdogMIS,           0xFC880018,__READ       ,__wdogmis_bits);
__IO_REG32(    WdogLock,          0xFC880C00,__READ_WRITE );
__IO_REG32(    WdogPeriphID0,     0xFC880FE0,__READ       );
__IO_REG32(    WdogPeriphID1,     0xFC880FE4,__READ       );
__IO_REG32(    WdogPeriphID2,     0xFC880FE8,__READ       );
__IO_REG32(    WdogPeriphID3,     0xFC880FEC,__READ       );
__IO_REG32(    WdogPCellID0,      0xFC880FF0,__READ       );
__IO_REG32(    WdogPCellID1,      0xFC880FF4,__READ       );
__IO_REG32(    WdogPCellID2,      0xFC880FF8,__READ       );
__IO_REG32(    WdogPCellID3,      0xFC880FFC,__READ       );

/***************************************************************************
 **
 **  CPU GPT0
 **
 ***************************************************************************/
__IO_REG16_BIT(TIMER0_CONTROL1,       0xF0000080,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMER0_STATUS_INT_ACK1,0xF0000084,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMER0_COMPARE1,       0xF0000088,__READ_WRITE );
__IO_REG16(    TIMER0_COUNT1,         0xF000008C,__READ       );
__IO_REG16(    TIMER0_REDG_CAPT1,     0xF0000090,__READ       );
__IO_REG16(    TIMER0_FEDG_CAPT1,     0xF0000094,__READ       );
__IO_REG16_BIT(TIMER0_CONTROL2,       0xF0000100,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMER0_STATUS_INT_ACK2,0xF0000104,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMER0_COMPARE2,       0xF0000108,__READ_WRITE );
__IO_REG16(    TIMER0_COUNT2,         0xF000010C,__READ       );
__IO_REG16(    TIMER0_REDG_CAPT2,     0xF0000110,__READ       );
__IO_REG16(    TIMER0_FEDG_CAPT2,     0xF0000114,__READ       );

/***************************************************************************
 **
 **  BS GPT1
 **
 ***************************************************************************/
__IO_REG16_BIT(TIMER1_CONTROL1,       0xFC800080,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMER1_STATUS_INT_ACK1,0xFC800084,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMER1_COMPARE1,       0xFC800088,__READ_WRITE );
__IO_REG16(    TIMER1_COUNT1,         0xFC80008C,__READ       );
__IO_REG16(    TIMER1_REDG_CAPT1,     0xFC800090,__READ       );
__IO_REG16(    TIMER1_FEDG_CAPT1,     0xFC800094,__READ       );
__IO_REG16_BIT(TIMER1_CONTROL2,       0xFC800100,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMER1_STATUS_INT_ACK2,0xFC800104,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMER1_COMPARE2,       0xFC800108,__READ_WRITE );
__IO_REG16(    TIMER1_COUNT2,         0xFC80010C,__READ       );
__IO_REG16(    TIMER1_REDG_CAPT2,     0xFC800110,__READ       );
__IO_REG16(    TIMER1_FEDG_CAPT2,     0xFC800114,__READ       );

/***************************************************************************
 **
 **  BS GPT2
 **
 ***************************************************************************/
__IO_REG16_BIT(TIMER2_CONTROL1,       0xFCB00080,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMER2_STATUS_INT_ACK1,0xFCB00084,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMER2_COMPARE1,       0xFCB00088,__READ_WRITE );
__IO_REG16(    TIMER2_COUNT1,         0xFCB0008C,__READ       );
__IO_REG16(    TIMER2_REDG_CAPT1,     0xFCB00090,__READ       );
__IO_REG16(    TIMER2_FEDG_CAPT1,     0xFCB00094,__READ       );
__IO_REG16_BIT(TIMER2_CONTROL2,       0xFCB00100,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMER2_STATUS_INT_ACK2,0xFCB00104,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMER2_COMPARE2,       0xFCB00108,__READ_WRITE );
__IO_REG16(    TIMER2_COUNT2,         0xFCB0010C,__READ       );
__IO_REG16(    TIMER2_REDG_CAPT2,     0xFCB00110,__READ       );
__IO_REG16(    TIMER2_FEDG_CAPT2,     0xFCB00114,__READ       );

/***************************************************************************
 **
 **  BS GPIO
 **
 ***************************************************************************/
__IO_REG16_BIT(GPIODATA,              0xFC9800FC,__READ_WRITE ,__gpiodata_bits);
__IO_REG16_BIT(GPIODIR,               0xFC980400,__READ_WRITE ,__gpiodir_bits);
__IO_REG16_BIT(GPIOIS,                0xFC980404,__READ_WRITE ,__gpiois_bits);
__IO_REG16_BIT(GPIOIBE,               0xFC980408,__READ_WRITE ,__gpioibe_bits);
__IO_REG16_BIT(GPIOIEV,               0xFC98040C,__READ_WRITE ,__gpioiev_bits);
__IO_REG16_BIT(GPIOIE,                0xFC980410,__READ_WRITE ,__gpioie_bits);
__IO_REG16_BIT(GPIORIS,               0xFC980414,__READ       ,__gpioris_bits);
__IO_REG16_BIT(GPIOMIS,               0xFC980418,__READ       ,__gpiomis_bits);
__IO_REG16_BIT(GPIOIC,                0xFC98041C,__WRITE      ,__gpioic_bits);
__IO_REG16(    GPIOPeriphID0,         0xFC980FE0,__READ       );
__IO_REG16(    GPIOPeriphID1,         0xFC980FE4,__READ       );
__IO_REG16(    GPIOPeriphID2,         0xFC980FE8,__READ       );
__IO_REG16(    GPIOPeriphID3,         0xFC980FEC,__READ       );
__IO_REG16(    GPIOPCellID0,          0xFC980FF0,__READ       );
__IO_REG16(    GPIOPCellID1,          0xFC980FF4,__READ       );
__IO_REG16(    GPIOPCellID2,          0xFC980FF8,__READ       );
__IO_REG16(    GPIOPCellID3,          0xFC980FFC,__READ       );

/***************************************************************************
 **
 **  DMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACIntStatus,         0xFC400000,__READ       ,__dmacintstatus_bits);
__IO_REG32_BIT(DMACIntTCStatus,       0xFC400004,__READ       ,__dmacinttcstatus_bits);
__IO_REG32_BIT(DMACIntTCClear,        0xFC400008,__WRITE      ,__dmacinttcclear_bits);
__IO_REG32_BIT(DMACIntErrorStatus,    0xFC40000C,__READ       ,__dmacinterrorstatus_bits);
__IO_REG32_BIT(DMACIntErrClr,         0xFC400010,__WRITE      ,__dmacinterrclr_bits);
__IO_REG32_BIT(DMACRawIntTCStatus,    0xFC400014,__READ       ,__dmacrawinttcstatus_bits);
__IO_REG32_BIT(DMACRawIntErrorStatus, 0xFC400018,__READ       ,__dmacrawinterrorstatus_bits);
__IO_REG32_BIT(DMACEnbldChns,         0xFC40001C,__READ       ,__dmacenbldchns_bits);
__IO_REG32_BIT(DMACSoftBReq,          0xFC400020,__READ_WRITE ,__dmacsoftbreq_bits);
__IO_REG32_BIT(DMACSoftSReq,          0xFC400024,__READ_WRITE ,__dmacsoftsreq_bits);
__IO_REG32_BIT(DMACSoftLBReq,         0xFC400028,__READ_WRITE ,__dmacsoftlbreq_bits);
__IO_REG32_BIT(DMACSoftLSReq,         0xFC40002C,__READ_WRITE ,__dmacsoftlsreq_bits);
__IO_REG32_BIT(DMACConfiguration,     0xFC400030,__READ_WRITE ,__dmacconfiguration_bits);
__IO_REG32_BIT(DMACSync,              0xFC400034,__READ_WRITE ,__dmacsync_bits);
__IO_REG32(    DMACC0SrcAddr,         0xFC400100,__READ_WRITE );
__IO_REG32(    DMACC0DestAddr,        0xFC400104,__READ_WRITE );
__IO_REG32_BIT(DMACC0LLI,             0xFC400108,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC0Control,         0xFC40010C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC0Configuration,   0xFC400110,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC1SrcAddr,         0xFC400120,__READ_WRITE );
__IO_REG32(    DMACC1DestAddr,        0xFC400124,__READ_WRITE );
__IO_REG32_BIT(DMACC1LLI,             0xFC400128,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC1Control,         0xFC40012C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC1Configuration,   0xFC400130,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC2SrcAddr,         0xFC400140,__READ_WRITE );
__IO_REG32(    DMACC2DestAddr,        0xFC400144,__READ_WRITE );
__IO_REG32_BIT(DMACC2LLI,             0xFC400148,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC2Control,         0xFC40014C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC2Configuration,   0xFC400150,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC3SrcAddr,         0xFC400160,__READ_WRITE );
__IO_REG32(    DMACC3DestAddr,        0xFC400164,__READ_WRITE );
__IO_REG32_BIT(DMACC3LLI,             0xFC400168,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC3Control,         0xFC40016C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC3Configuration,   0xFC400170,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC4SrcAddr,         0xFC400180,__READ_WRITE );
__IO_REG32(    DMACC4DestAddr,        0xFC400184,__READ_WRITE );
__IO_REG32_BIT(DMACC4LLI,             0xFC400188,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC4Control,         0xFC40018C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC4Configuration,   0xFC400190,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC5SrcAddr,         0xFC4001A0,__READ_WRITE );
__IO_REG32(    DMACC5DestAddr,        0xFC4001A4,__READ_WRITE );
__IO_REG32_BIT(DMACC5LLI,             0xFC4001A8,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC5Control,         0xFC4001AC,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC5Configuration,   0xFC4001B0,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC6SrcAddr,         0xFC4001C0,__READ_WRITE );
__IO_REG32(    DMACC6DestAddr,        0xFC4001C4,__READ_WRITE );
__IO_REG32_BIT(DMACC6LLI,             0xFC4001C8,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC6Control,         0xFC4001CC,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC6Configuration,   0xFC4001D0,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC7SrcAddr,         0xFC4001E0,__READ_WRITE );
__IO_REG32(    DMACC7DestAddr,        0xFC4001E4,__READ_WRITE );
__IO_REG32_BIT(DMACC7LLI,             0xFC4001E8,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC7Control,         0xFC4001EC,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC7Configuration,   0xFC4001F0,__READ_WRITE ,__dmaccconfiguration_bits);

/***************************************************************************
 **
 **  RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTCTIME,               0xFC900000,__READ_WRITE ,__rtctime_bits);
__IO_REG32_BIT(RTCDATE,               0xFC900004,__READ_WRITE ,__rtcdate_bits);
__IO_REG32_BIT(RTCALARMTIME,          0xFC900008,__READ_WRITE ,__rtctime_bits);
__IO_REG32_BIT(RTCALARMDATE,          0xFC90000C,__READ_WRITE ,__rtcdate_bits);
__IO_REG32_BIT(RTCCONTROL,            0xFC900010,__READ_WRITE ,__rtccontrol_bits);
__IO_REG32_BIT(RTCSTATUS,             0xFC900014,__READ_WRITE ,__rtcstatus_bits);
__IO_REG32(    RTCREG1MC,             0xFC900018,__READ_WRITE );
__IO_REG32(    RTCREG2MC,             0xFC90001C,__READ_WRITE );

#if 0
/***************************************************************************
 **
 **  C3
 **
 ***************************************************************************/
__IO_REG32_BIT(C3_SYS_SCR,            0xD9000000,__READ_WRITE ,__c3_sys_scr_bits);
__IO_REG32_BIT(C3_SYS_STR,            0xD9000040,__READ       ,__c3_sys_scr_bits);
__IO_REG32_BIT(C3_SYS_VER,            0xD90003F0,__READ       ,__c3_sys_ver_bits);
__IO_REG32(    C3_SYS_HWID,           0xD90003FC,__READ       );
__IO_REG32_BIT(C3_HIF_MP_BASE,        0xD9000400,__READ_WRITE );
__IO_REG32_BIT(C3_HIF_MSIZE,          0xD9000700,__READ       ,__c3_hif_msize_bits);
__IO_REG32(    C3_HIF_MBAR,           0xD9000704,__READ_WRITE );
__IO_REG32_BIT(C3_HIF_MCAR,           0xD9000708,__READ_WRITE ,__c3_hif_mcar_bits);
__IO_REG32(    C3_HIF_MPBAR,          0xD900070C,__READ_WRITE );
__IO_REG32_BIT(C3_HIF_MAAR,           0xD9000710,__READ_WRITE ,__c3_hif_maar_bits);
__IO_REG32(    C3_HIF_MADR,           0xD9000714,__READ_WRITE );
__IO_REG32(    C3_HIF_NBAR,           0xD9000744,__READ_WRITE );
__IO_REG32_BIT(C3_HIF_NCR,            0xD9000748,__READ_WRITE ,__c3_hif_ncr_bits);
__IO_REG32_BIT(C3_ID_SCR,             0xD9001000,__READ_WRITE ,__c3_id_scr_bits);
__IO_REG32_BIT(C3_ID_IP,              0xD9001010,__READ_WRITE ,__c3_id_ip_bits);
__IO_REG32_BIT(C3_ID_IR0,             0xD9001020,__READ_WRITE ,__c3_id_ir0_bits);
__IO_REG32_BIT(C3_ID_IR1,             0xD9001024,__READ_WRITE ,__c3_id_ir1_bits);
__IO_REG32_BIT(C3_ID_IR2,             0xD9001028,__READ_WRITE ,__c3_id_ir2_bits);
__IO_REG32_BIT(C3_ID_IR3,             0xD900102C,__READ_WRITE ,__c3_id_ir3_bits);
#endif

/***************************************************************************
 **
 **  EMI
 **
 ***************************************************************************/
__IO_REG8(     tSCS_0_reg,            0x4F000000,__READ_WRITE );
__IO_REG8(     tSE_0_reg,             0x4F000004,__READ_WRITE );
__IO_REG8(     tENw_0_reg,            0x4F000008,__READ_WRITE );
__IO_REG8(     tENr_0_reg,            0x4F00000C,__READ_WRITE );
__IO_REG8(     tDCS_0_reg,            0x4F000010,__READ_WRITE );
__IO_REG8_BIT( control_0_reg,         0x4F000014,__READ_WRITE ,__control_reg_bits);
__IO_REG8(     tSCS_1_reg,            0x4F000018,__READ_WRITE );
__IO_REG8(     tSE_1_reg,             0x4F00001C,__READ_WRITE );
__IO_REG8(     tENw_1_reg,            0x4F000020,__READ_WRITE );
__IO_REG8(     tENr_1_reg,            0x4F000024,__READ_WRITE );
__IO_REG8(     tDCS_1_reg,            0x4F000028,__READ_WRITE );
__IO_REG8_BIT( control_1_reg,         0x4F00002C,__READ_WRITE ,__control_reg_bits);
__IO_REG8(     tSCS_2_reg,            0x4F000030,__READ_WRITE );
__IO_REG8(     tSE_2_reg,             0x4F000034,__READ_WRITE );
__IO_REG8(     tENw_2_reg,            0x4F000038,__READ_WRITE );
__IO_REG8(     tENr_2_reg,            0x4F00003C,__READ_WRITE );
__IO_REG8(     tDCS_2_reg,            0x4F000040,__READ_WRITE );
__IO_REG8_BIT( control_2_reg,         0x4F000044,__READ_WRITE ,__control_reg_bits);
__IO_REG8(     tSCS_3_reg,            0x4F000048,__READ_WRITE );
__IO_REG8(     tSE_3_reg,             0x4F00004C,__READ_WRITE );
__IO_REG8(     tENw_3_reg,            0x4F000050,__READ_WRITE );
__IO_REG8(     tENr_3_reg,            0x4F000054,__READ_WRITE );
__IO_REG8(     tDCS_3_reg,            0x4F000058,__READ_WRITE );
__IO_REG8_BIT( control_3_reg,         0x4F00005C,__READ_WRITE ,__control_reg_bits);
__IO_REG8(     tSCS_4_reg,            0x4F000060,__READ_WRITE );
__IO_REG8(     tSE_4_reg,             0x4F000064,__READ_WRITE );
__IO_REG8(     tENw_4_reg,            0x4F000068,__READ_WRITE );
__IO_REG8(     tENr_4_reg,            0x4F00006C,__READ_WRITE );
__IO_REG8(     tDCS_4_reg,            0x4F000070,__READ_WRITE );
__IO_REG8_BIT( control_4_reg,         0x4F000074,__READ_WRITE ,__control_reg_bits);
__IO_REG8(     tSCS_5_reg,            0x4F000078,__READ_WRITE );
__IO_REG8(     tSE_5_reg,             0x4F00007C,__READ_WRITE );
__IO_REG8(     tENw_5_reg,            0x4F000080,__READ_WRITE );
__IO_REG8(     tENr_5_reg,            0x4F000084,__READ_WRITE );
__IO_REG8(     tDCS_5_reg,            0x4F000088,__READ_WRITE );
__IO_REG8_BIT( control_5_reg,         0x4F00008C,__READ_WRITE ,__control_reg_bits);
__IO_REG8(     timeout_reg,           0x4F000090,__READ_WRITE );
__IO_REG8_BIT( ack_reg,               0x4F000094,__READ_WRITE ,__ack_reg_bits);
__IO_REG8_BIT( irq_reg,               0x4F000098,__READ_WRITE ,__irq_reg_bits);

/***************************************************************************
 **
 **  EHCI
 **
 ***************************************************************************/
__IO_REG32_BIT(HCCAPBASE,             0xE1800000,__READ       ,__hccapbase_bits);
__IO_REG32_BIT(HCSPARAMS,             0xE1800004,__READ       ,__hcsparams_bits);
__IO_REG32_BIT(HCCPARAMS,             0xE1800008,__READ       ,__hccparams_bits);
__IO_REG32_BIT(HCUSBCMD,              0xE1800010,__READ       ,__hcusbcmd_bits);
__IO_REG32_BIT(HCUSBSTS,              0xE1800014,__READ_WRITE ,__hcusbsts_bits);
__IO_REG32_BIT(HCUSBINTR,             0xE1800018,__READ_WRITE ,__hcusbintr_bits);
__IO_REG32_BIT(HCFRINDEX,             0xE180001C,__READ_WRITE ,__hcfrindex_bits);
__IO_REG32(    HCCTRLDSSEGMENT,       0xE1800020,__READ_WRITE );
__IO_REG32(    HCPERIODICLISTBASE,    0xE1800024,__READ_WRITE );
__IO_REG32(    HCASYNCLISTADDR,       0xE1800028,__READ_WRITE );
__IO_REG32_BIT(HCCONFIGFLAG,          0xE1800050,__READ_WRITE ,__hcconfigflag_bits);
__IO_REG32_BIT(HCPORTSC1,             0xE1800054,__READ_WRITE ,__hcportsc_bits);
__IO_REG32_BIT(HCPORTSC2,             0xE1800058,__READ_WRITE ,__hcportsc_bits);
__IO_REG32_BIT(HCINSNREG00,           0xE1800090,__READ_WRITE ,__hcinsnreg00_bits);
__IO_REG32_BIT(HCINSNREG01,           0xE1800094,__READ_WRITE ,__hcinsnreg01_bits);
__IO_REG32_BIT(HCINSNREG02,           0xE1800098,__READ_WRITE ,__hcinsnreg02_bits);
__IO_REG32_BIT(HCINSNREG03,           0xE180009C,__READ_WRITE ,__hcinsnreg03_bits);
__IO_REG32_BIT(HCINSNREG05,           0xE18000A4,__READ_WRITE ,__hcinsnreg05_bits);

/***************************************************************************
 **
 **  OCHI0
 **
 ***************************************************************************/
__IO_REG32_BIT(Hc0Revision,           0xE1900000,__READ       ,__hcrevision_bits);
__IO_REG32_BIT(Hc0Control,            0xE1900004,__READ_WRITE ,__hccontrol_bits);
__IO_REG32_BIT(Hc0CommandStatus,      0xE1900008,__READ_WRITE ,__hccommandstatus_bits);
__IO_REG32_BIT(Hc0InterruptStatus,    0xE190000C,__READ_WRITE ,__hcinterruptstatus_bits);
__IO_REG32_BIT(Hc0InterruptEnable,    0xE1900010,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(Hc0InterruptDisable,   0xE1900014,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(Hc0HCCA,               0xE1900018,__READ_WRITE ,__hchcca_bits);
__IO_REG32_BIT(Hc0PeriodCurrentED,    0xE190001C,__READ       ,__hcperiodcurrented_bits);
__IO_REG32_BIT(Hc0ControlHeadED,      0xE1900020,__READ_WRITE ,__hccontrolheaded_bits);
__IO_REG32_BIT(Hc0ControlCurrentED,   0xE1900024,__READ_WRITE ,__hccontrolcurrented_bits);
__IO_REG32_BIT(Hc0BulkHeadED,         0xE1900028,__READ_WRITE ,__hcbulkheaded_bits);
__IO_REG32_BIT(Hc0BulkCurrentED,      0xE190002C,__READ_WRITE ,__hcbulkcurrented_bits);
__IO_REG32_BIT(Hc0DoneHead,           0xE1900030,__READ       ,__hcdonehead_bits);
__IO_REG32_BIT(Hc0FmInterval,         0xE1900034,__READ_WRITE ,__hcfminterval_bits);
__IO_REG32_BIT(Hc0FmRemaining,        0xE1900038,__READ       ,__hcfmremaining_bits);
__IO_REG32_BIT(Hc0FmNumber,           0xE190003C,__READ       ,__hcfmnumber_bits);
__IO_REG32_BIT(Hc0PeriodStart,        0xE1900040,__READ_WRITE ,__hcperiodicstart_bits);
__IO_REG32_BIT(Hc0LSThreshold,        0xE1900044,__READ_WRITE ,__hclsthreshold_bits);
__IO_REG32_BIT(Hc0RhDescriptorA,      0xE1900048,__READ_WRITE ,__hcrhdescriptora_bits);
__IO_REG32_BIT(Hc0RhDescripterB,      0xE190004C,__READ_WRITE ,__hcrhdescriptorb_bits);
__IO_REG32_BIT(Hc0RhStatus,           0xE1900050,__READ_WRITE ,__hcrhstatus_bits);
__IO_REG32_BIT(Hc0RhPortStatus,       0xE1900054,__READ_WRITE ,__hcrhportstatus_bits);

/***************************************************************************
 **
 **  OCHI1
 **
 ***************************************************************************/
__IO_REG32_BIT(Hc1Revision,           0xE2100000,__READ       ,__hcrevision_bits);
__IO_REG32_BIT(Hc1Control,            0xE2100004,__READ_WRITE ,__hccontrol_bits);
__IO_REG32_BIT(Hc1CommandStatus,      0xE2100008,__READ_WRITE ,__hccommandstatus_bits);
__IO_REG32_BIT(Hc1InterruptStatus,    0xE210000C,__READ_WRITE ,__hcinterruptstatus_bits);
__IO_REG32_BIT(Hc1InterruptEnable,    0xE2100010,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(Hc1InterruptDisable,   0xE2100014,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(Hc1HCCA,               0xE2100018,__READ_WRITE ,__hchcca_bits);
__IO_REG32_BIT(Hc1PeriodCurrentED,    0xE210001C,__READ       ,__hcperiodcurrented_bits);
__IO_REG32_BIT(Hc1ControlHeadED,      0xE2100020,__READ_WRITE ,__hccontrolheaded_bits);
__IO_REG32_BIT(Hc1ControlCurrentED,   0xE2100024,__READ_WRITE ,__hccontrolcurrented_bits);
__IO_REG32_BIT(Hc1BulkHeadED,         0xE2100028,__READ_WRITE ,__hcbulkheaded_bits);
__IO_REG32_BIT(Hc1BulkCurrentED,      0xE210002C,__READ_WRITE ,__hcbulkcurrented_bits);
__IO_REG32_BIT(Hc1DoneHead,           0xE2100030,__READ       ,__hcdonehead_bits);
__IO_REG32_BIT(Hc1FmInterval,         0xE2100034,__READ_WRITE ,__hcfminterval_bits);
__IO_REG32_BIT(Hc1FmRemaining,        0xE2100038,__READ       ,__hcfmremaining_bits);
__IO_REG32_BIT(Hc1FmNumber,           0xE210003C,__READ       ,__hcfmnumber_bits);
__IO_REG32_BIT(Hc1PeriodStart,        0xE2100040,__READ_WRITE ,__hcperiodicstart_bits);
__IO_REG32_BIT(Hc1LSThreshold,        0xE2100044,__READ_WRITE ,__hclsthreshold_bits);
__IO_REG32_BIT(Hc1RhDescriptorA,      0xE2100048,__READ_WRITE ,__hcrhdescriptora_bits);
__IO_REG32_BIT(Hc1RhDescripterB,      0xE210004C,__READ_WRITE ,__hcrhdescriptorb_bits);
__IO_REG32_BIT(Hc1RhStatus,           0xE2100050,__READ_WRITE ,__hcrhstatus_bits);
__IO_REG32_BIT(Hc1RhPortStatus,       0xE2100054,__READ_WRITE ,__hcrhportstatus_bits);

/***************************************************************************
 **
 **  USBD
 **
 ***************************************************************************/
__IO_REG32_BIT(UDEP0INCTRL,           0xE1100000,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP0INSTAT,           0xE1100004,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP0INBS,             0xE1100008,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP0INMPS,            0xE110000C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP0INDDP,            0xE1100014,__READ_WRITE );
__IO_REG32(    UDEP0INWC,             0xE1100018,__WRITE      );
__IO_REG32_BIT(UDEP1INCTRL,           0xE1100020,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP1INSTAT,           0xE1100024,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP1INBS,             0xE1100028,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP1INMPS,            0xE110002C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP1INDDP,            0xE1100034,__READ_WRITE );
__IO_REG32(    UDEP1INWC,             0xE1100038,__WRITE    );
__IO_REG32_BIT(UDEP3INCTRL,           0xE1100060,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP3INSTAT,           0xE1100064,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP3INBS,             0xE1100068,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP3INMPS,            0xE110006C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP3INDDP,            0xE1100074,__READ_WRITE );
__IO_REG32(    UDEP3INWC,             0xE1100078,__WRITE      );
__IO_REG32_BIT(UDEP5INCTRL,           0xE11000A0,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP5INSTAT,           0xE11000A4,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP5INBS,             0xE11000A8,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP5INMPS,            0xE11000AC,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP5INDDP,            0xE11000B4,__READ_WRITE );
__IO_REG32(    UDEP5INWC,             0xE11000B8,__WRITE      );
__IO_REG32_BIT(UDEP7INCTRL,           0xE11000E0,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP7INSTAT,           0xE11000E4,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP7INBS,             0xE11000E8,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP7INMPS,            0xE11000EC,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP7INDDP,            0xE11000F4,__READ_WRITE );
__IO_REG32(    UDEP7INWC,             0xE11000F8,__WRITE      );
__IO_REG32_BIT(UDEP9INCTRL,           0xE1100120,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP9INSTAT,           0xE1100124,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP9INBS,             0xE1100128,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP9INMPS,            0xE110012C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP9INDDP,            0xE1100134,__READ_WRITE );
__IO_REG32(    UDEP9INWC,             0xE1100138,__WRITE      );
__IO_REG32_BIT(UDEP11INCTRL,          0xE1100160,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP11INSTAT,          0xE1100164,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP11INBS,            0xE1100168,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP11INMPS,           0xE110016C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP11INDDP,           0xE1100174,__READ_WRITE );
__IO_REG32(    UDEP11INWC,            0xE1100178,__WRITE      );
__IO_REG32_BIT(UDEP13INCTRL,          0xE11001A0,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP13INSTAT,          0xE11001A4,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP13INBS,            0xE11001A8,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP13INMPS,           0xE11001AC,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP13INDDP,           0xE11001B4,__READ_WRITE );
__IO_REG32(    UDEP13INWC,            0xE11001B8,__WRITE      );
__IO_REG32_BIT(UDEP15INCTRL,          0xE11001E0,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP15INSTAT,          0xE11001E4,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP15INBS,            0xE11001E8,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP15INMPS,           0xE11001EC,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP15INDDP,           0xE11001F4,__READ_WRITE );
__IO_REG32(    UDEP15INWC,            0xE11001F8,__WRITE      );
__IO_REG32_BIT(UDEP0OUTCTRL,          0xE1100200,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP0OUTSTAT,          0xE1100204,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP0OUTPFN,           0xE1100208,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP0OUTBS,            0xE110020C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP0OUSBP,            0xE1100210,__READ_WRITE );
__IO_REG32(    UDEP0OUTDDP,           0xE1100214,__READ_WRITE );
__IO_REG32(    UDEP0OUTRC,            0xE110021C,__READ       );
__IO_REG32_BIT(UDEP2OUTCTRL,          0xE1100240,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP2OUTSTAT,          0xE1100244,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP2OUTPFN,           0xE1100248,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP2OUTBS,            0xE110024C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP2OUSBP,            0xE1100250,__READ_WRITE );
__IO_REG32(    UDEP2OUTDDP,           0xE1100254,__READ_WRITE );
__IO_REG32(    UDEP2OUTRC,            0xE110025C,__READ       );
__IO_REG32_BIT(UDEP4OUTCTRL,          0xE1100280,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP4OUTSTAT,          0xE1100284,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP4OUTPFN,           0xE1100288,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP4OUTBS,            0xE110028C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP4OUSBP,            0xE1100290,__READ_WRITE );
__IO_REG32(    UDEP4OUTDDP,           0xE1100294,__READ_WRITE );
__IO_REG32(    UDEP4OUTRC,            0xE110029C,__READ       );
__IO_REG32_BIT(UDEP6OUTCTRL,          0xE11002C0,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP6OUTSTAT,          0xE11002C4,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP6OUTPFN,           0xE11002C8,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP6OUTBS,            0xE11002CC,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP6OUSBP,            0xE11002D0,__READ_WRITE );
__IO_REG32(    UDEP6OUTDDP,           0xE11002D4,__READ_WRITE );
__IO_REG32(    UDEP6OUTRC,            0xE11002DC,__READ       );
__IO_REG32_BIT(UDEP8OUTCTRL,          0xE1100300,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP8OUTSTAT,          0xE1100304,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP8OUTPFN,           0xE1100308,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP8OUTBS,            0xE110030C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP8OUSBP,            0xE1100310,__READ_WRITE );
__IO_REG32(    UDEP8OUTDDP,           0xE1100314,__READ_WRITE );
__IO_REG32(    UDEP8OUTRC,            0xE110031C,__READ       );
__IO_REG32_BIT(UDEP10OUTCTRL,         0xE1100340,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP10OUTSTAT,         0xE1100344,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP10OUTPFN,          0xE1100348,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP10OUTBS,           0xE110034C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP10OUSBP,           0xE1100350,__READ_WRITE );
__IO_REG32(    UDEP10OUTDDP,          0xE1100354,__READ_WRITE );
__IO_REG32(    UDEP10OUTRC,           0xE110035C,__READ       );
__IO_REG32_BIT(UDEP12OUTCTRL,         0xE1100380,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP12OUTSTAT,         0xE1100384,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP12OUTPFN,          0xE1100388,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP12OUTBS,           0xE110038C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP12OUSBP,           0xE1100390,__READ_WRITE );
__IO_REG32(    UDEP12OUTDDP,          0xE1100394,__READ_WRITE );
__IO_REG32(    UDEP12OUTRC,           0xE110039C,__READ       );
__IO_REG32_BIT(UDEP14OUTCTRL,         0xE11003C0,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP14OUTSTAT,         0xE11003C4,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP14OUTPFN,          0xE11003C8,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP14OUTBS,           0xE11003CC,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP14OUSBP,           0xE11003D0,__READ_WRITE );
__IO_REG32(    UDEP14OUTDDP,          0xE11003D4,__READ_WRITE );
__IO_REG32(    UDEP14OUTRC,           0xE11003DC,__READ       );
__IO_REG32_BIT(UDDCFG,                0xE1100400,__READ_WRITE ,__uddcfg_bits);
__IO_REG32_BIT(UDDCTRL,               0xE1100404,__READ_WRITE ,__uddctrl_bits);
__IO_REG32_BIT(UDDSTAT,               0xE1100408,__READ       ,__uddstat_bits);
__IO_REG32_BIT(UDDINTR,               0xE110040C,__READ_WRITE ,__uddintr_bits);
__IO_REG32_BIT(UDDIM,                 0xE1100410,__READ_WRITE ,__uddintr_bits);
__IO_REG32_BIT(UDEINTR,               0xE1100414,__READ_WRITE ,__udeintr_bits);
__IO_REG32_BIT(UDEIM,                 0xE1100418,__READ_WRITE ,__udeintr_bits);
__IO_REG32_BIT(UDEP0,                 0xE1100504,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP1,                 0xE1100508,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP2,                 0xE110050C,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP3,                 0xE1100510,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP4,                 0xE1100514,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP5,                 0xE1100518,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP6,                 0xE110051C,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP7,                 0xE1100520,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP8,                 0xE1100524,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP9,                 0xE1100528,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP10,                0xE110052C,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP11,                0xE1100530,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP12,                0xE1100534,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP13,                0xE1100538,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP14,                0xE110053C,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP15,                0xE1100540,__READ_WRITE ,__udep_bits);

/***************************************************************************
 **
 **  MII-DMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA_BMR,               0xE0801000,__READ_WRITE ,__dma_bmr_bits);
__IO_REG32(    DMA_TPDR,              0xE0801004,__READ_WRITE );
__IO_REG32(    DMA_RPDR,              0xE0801008,__READ_WRITE );
__IO_REG32(    DMA_RDLAR,             0xE080100C,__READ_WRITE );
__IO_REG32(    DMA_TDLAR,             0xE0801010,__READ_WRITE );
__IO_REG32_BIT(DMA_SR,                0xE0801014,__READ_WRITE ,__dma_sr_bits);
__IO_REG32_BIT(DMA_OMR,               0xE0801018,__READ_WRITE ,__dma_omr_bits);
__IO_REG32_BIT(DMA_IER,               0xE080101C,__READ_WRITE ,__dma_ier_bits);
__IO_REG32_BIT(DMA_MFABOCR,           0xE0801020,__READ_WRITE ,__dma_mfabocr_bits);
__IO_REG32(    DMA_CHTDR,             0xE0801048,__READ       );
__IO_REG32(    DMA_CHRDR,             0xE080104C,__READ       );
__IO_REG32(    DMA_CHTBAR,            0xE0801050,__READ       );
__IO_REG32(    DMA_CHRBAR,            0xE0801054,__READ       );

/***************************************************************************
 **
 **  MII-MAC
 **
 ***************************************************************************/
__IO_REG32_BIT(MAC_CR,                0xE0800000,__READ_WRITE ,__mac_cr_bits);
__IO_REG32_BIT(MAC_FFR,               0xE0800004,__READ_WRITE ,__mac_ffr_bits);
__IO_REG32(    MAC_HTHR,              0xE0800008,__READ_WRITE );
__IO_REG32(    MAC_HTLR,              0xE080000C,__READ_WRITE );
__IO_REG32_BIT(MAC_MIIAR,             0xE0800010,__READ_WRITE ,__mac_miiar_bits);
__IO_REG32_BIT(MAC_MIIDR,             0xE0800014,__READ_WRITE ,__mac_miidr_bits);
__IO_REG32_BIT(MAC_FCR,               0xE0800018,__READ_WRITE ,__mac_fcr_bits);
__IO_REG32_BIT(MAC_VLANTR,            0xE080001C,__READ_WRITE ,__mac_vlantr_bits);
__IO_REG32_BIT(MAC_VR,                0xE0800020,__READ       ,__mac_vr_bits);
__IO_REG32(    MAC_PTWUFFR,           0xE0800028,__READ_WRITE );
__IO_REG32_BIT(MAC_PMTCASR,           0xE080002C,__READ_WRITE ,__mac_pmtcasr_bits);
__IO_REG32_BIT(MAC_IR,                0xE0800038,__READ_WRITE ,__mac_ir_bits);
__IO_REG32_BIT(MAC_IMR,               0xE080003C,__READ_WRITE ,__mac_imr_bits);
__IO_REG32_BIT(MAC_A0HR,              0xE0800040,__READ_WRITE ,__mac_ahr0_bits);
__IO_REG32(    MAC_A0LR,              0xE0800044,__READ_WRITE );
__IO_REG32_BIT(MAC_A1HR,              0xE0800048,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A1LR,              0xE080004C,__READ_WRITE );
__IO_REG32_BIT(MAC_A2HR,              0xE0800050,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A2LR,              0xE0800054,__READ_WRITE );
__IO_REG32_BIT(MAC_A3HR,              0xE0800058,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A3LR,              0xE080005C,__READ_WRITE );
__IO_REG32_BIT(MAC_A4HR,              0xE0800060,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A4LR,              0xE0800064,__READ_WRITE );
__IO_REG32_BIT(MAC_A5HR,              0xE0800068,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A5LR,              0xE080006C,__READ_WRITE );
__IO_REG32_BIT(MAC_A6HR,              0xE0800070,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A6LR,              0xE0800074,__READ_WRITE );
__IO_REG32_BIT(MAC_A7HR,              0xE0800078,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A7LR,              0xE080007C,__READ_WRITE );
__IO_REG32_BIT(MAC_A8HR,              0xE0800080,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A8LR,              0xE0800084,__READ_WRITE );
__IO_REG32_BIT(MAC_A9HR,              0xE0800088,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A9LR,              0xE080008C,__READ_WRITE );
__IO_REG32_BIT(MAC_A10HR,             0xE0800090,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A10LR,             0xE0800094,__READ_WRITE );
__IO_REG32_BIT(MAC_A11HR,             0xE0800098,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A11LR,             0xE080009C,__READ_WRITE );
__IO_REG32_BIT(MAC_A12HR,             0xE08000A0,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A12LR,             0xE08000A4,__READ_WRITE );
__IO_REG32_BIT(MAC_A13HR,             0xE08000A8,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A13LR,             0xE08000AC,__READ_WRITE );
__IO_REG32_BIT(MAC_A14HR,             0xE08000B0,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A14LR,             0xE08000B4,__READ_WRITE );
__IO_REG32_BIT(MAC_A15HR,             0xE08000B8,__READ_WRITE ,__mac_ahr_bits);
__IO_REG32(    MAC_A15LR,             0xE08000BC,__READ_WRITE );

/***************************************************************************
 **
 **  MII-MMC
 **
 ***************************************************************************/
__IO_REG32_BIT(Mmc_cntrl,             0xE0800100,__READ_WRITE ,__mmc_cntrl_bits);
__IO_REG32_BIT(Mmc_intr_rx,           0xE0800104,__READ_WRITE ,__mmc_intr_rx_bits);
__IO_REG32_BIT(Mmc_intr_tx,           0xE0800108,__READ_WRITE ,__mmc_intr_tx_bits);
__IO_REG32_BIT(Mmc_intr_mask_rx,      0xE080010C,__READ_WRITE ,__mmc_intr_mask_rx_bits);
__IO_REG32_BIT(Mmc_intr_mask_tx,      0xE0800110,__READ_WRITE ,__mmc_intr_mask_tx_bits);
__IO_REG32(    Txoctetcount_gb,       0xE0800114,__READ_WRITE );
__IO_REG32(    Txframecount_gb,       0xE0800118,__READ_WRITE );
__IO_REG32(    Txbroadcastframes_g,   0xE080011C,__READ_WRITE );
__IO_REG32(    Txmulticastframes_g,   0xE0800120,__READ_WRITE );
__IO_REG32(    Tx64octects_gb,        0xE0800124,__READ_WRITE );
__IO_REG32(    Tx65to127octects_gb,   0xE0800128,__READ_WRITE );
__IO_REG32(    Tx128to255octects_gb,  0xE080012C,__READ_WRITE );
__IO_REG32(    Tx256to511octects_gb,  0xE0800130,__READ_WRITE );
__IO_REG32(    Tx512to1023octects_gb, 0xE0800134,__READ_WRITE );
__IO_REG32(    Tx1024tomaxoctects_gb, 0xE0800138,__READ_WRITE );
__IO_REG32(    Txunicastframes_gb,    0xE080013C,__READ_WRITE );
__IO_REG32(    Txmulticastframes_gb,  0xE0800140,__READ_WRITE );
__IO_REG32(    Txbroadcastframes_gb,  0xE0800144,__READ_WRITE );
__IO_REG32(    Txunderflowerror,      0xE0800148,__READ_WRITE );
__IO_REG32(    Txsinglecol_g,         0xE080014C,__READ_WRITE );
__IO_REG32(    Txmulticol_g,          0xE0800150,__READ_WRITE );
__IO_REG32(    Txdeferred,            0xE0800154,__READ_WRITE );
__IO_REG32(    Txlatecol,             0xE0800158,__READ_WRITE );
__IO_REG32(    Txexesscol,            0xE080015C,__READ_WRITE );
__IO_REG32(    Txcarriererror,        0xE0800160,__READ_WRITE );
__IO_REG32(    Txoctetcount_g,        0xE0800164,__READ_WRITE );
__IO_REG32(    Txexcessdef,           0xE0800168,__READ_WRITE );
__IO_REG32(    Txpauseframes,         0xE0800170,__READ_WRITE );
__IO_REG32(    Txvlanframes_g,        0xE0800174,__READ_WRITE );
__IO_REG32(    Rxframecount_gb,       0xE0800180,__READ_WRITE );
__IO_REG32(    Rxoctetcount_gb,       0xE0800184,__READ_WRITE );
__IO_REG32(    Rxoctetcount_g,        0xE0800188,__READ_WRITE );
__IO_REG32(    Rxbroadcastframes_g,   0xE080018C,__READ_WRITE );
__IO_REG32(    Rxmulticastframes_g,   0xE0800190,__READ_WRITE );
__IO_REG32(    Rxcrcerror,            0xE0800194,__READ_WRITE );
__IO_REG32(    Rxalignmenterror,      0xE0800198,__READ_WRITE );
__IO_REG32(    Rxrunterror,           0xE080019C,__READ_WRITE );
__IO_REG32(    Rxjabbererror,         0xE08001A0,__READ_WRITE );
__IO_REG32(    Rxundersize_g,         0xE08001A4,__READ_WRITE );
__IO_REG32(    Rxoversize_g,          0xE08001A8,__READ_WRITE );
__IO_REG32(    Rx64octects_gb,        0xE08001AC,__READ_WRITE );
__IO_REG32(    Rx65to127octects_gb,   0xE08001B0,__READ_WRITE );
__IO_REG32(    Rx128to255octects_gb,  0xE08001B4,__READ_WRITE );
__IO_REG32(    Rx256to511octects_gb,  0xE08001B8,__READ_WRITE );
__IO_REG32(    Rx512to1023octects_gb, 0xE08001BC,__READ_WRITE );
__IO_REG32(    Rx1023tomaxoctects_gb, 0xE08001C0,__READ_WRITE );
__IO_REG32(    Rxunicastframes_g,     0xE08001C4,__READ_WRITE );
__IO_REG32(    Rxlengtherror,         0xE08001C8,__READ_WRITE );
__IO_REG32(    Rxoutofrangetype,      0xE08001CC,__READ_WRITE );
__IO_REG32(    Rxpauseframes,         0xE08001D0,__READ_WRITE );
__IO_REG32(    Rxfifooverflow,        0xE08001D4,__READ_WRITE );
__IO_REG32(    Rxvlanframes_gb,       0xE08001D8,__READ_WRITE );
__IO_REG32(    Rxwatchdogerror,       0xE08001DC,__READ_WRITE );

/***************************************************************************
 **
 **  JPEG
 **
 ***************************************************************************/
__IO_REG32_BIT(JPGCReg0,              0xD0800000,__WRITE      ,__jpgcreg0_bits);
__IO_REG32_BIT(JPGCReg1,              0xD0800004,__READ_WRITE ,__jpgcreg1_bits);
__IO_REG32_BIT(JPGCReg2,              0xD0800008,__READ_WRITE ,__jpgcreg2_bits);
__IO_REG32_BIT(JPGCReg3,              0xD080000C,__READ_WRITE ,__jpgcreg3_bits);
__IO_REG32_BIT(JPGCReg4,              0xD0800010,__READ_WRITE ,__jpgcreg4_bits);
__IO_REG32_BIT(JPGCReg5,              0xD0800014,__READ_WRITE ,__jpgcreg4_bits);
__IO_REG32_BIT(JPGCReg6,              0xD0800018,__READ_WRITE ,__jpgcreg4_bits);
__IO_REG32_BIT(JPGCReg7,              0xD080001C,__READ_WRITE ,__jpgcreg4_bits);
__IO_REG32_BIT(JPGCCS,                0xD0800200,__READ_WRITE ,__jpgccs_bits);
__IO_REG32(    JPGCBFIFO2C,           0xD0800204,__READ       );
__IO_REG32(    JPGCBC2FIFO,           0xD0800208,__READ       );
__IO_REG32_BIT(JPGCBCBI,              0xD080020C,__READ_WRITE ,__jpgcbcbi_bits);
__IO_REG32(    JPGCFifoIn,            0xD0800400,__READ_WRITE );
__IO_REG32(    JPGCFifoOut,           0xD0800600,__READ_WRITE );
__IO_REG32(    JPGCQMem,              0xD0800800,__READ_WRITE );
__IO_REG32(    JPGCHuffMin,           0xD0800C00,__READ_WRITE );
__IO_REG32(    JPGCHuffBase,          0xD0801000,__READ_WRITE );
__IO_REG32(    JPGCHuffSymb,          0xD0801400,__READ_WRITE );
__IO_REG32(    JPGCDHTMem,            0xD0801800,__READ_WRITE );
__IO_REG32(    JPGCHuffEnc,           0xD0801C00,__READ_WRITE );

/***************************************************************************
 **
 **  FIrDA
 **
 ***************************************************************************/
__IO_REG32_BIT(IrDA_CON,              0xD1000010,__READ_WRITE ,__irda_con_bits);
__IO_REG32_BIT(IrDA_CONF,             0xD1000014,__READ_WRITE ,__irda_conf_bits);
__IO_REG32_BIT(IrDA_PARA,             0xD1000018,__READ_WRITE ,__irda_para_bits);
__IO_REG32_BIT(IrDA_DV,               0xD100001C,__READ_WRITE ,__irda_dv_bits);
__IO_REG32_BIT(IrDA_STAT,             0xD1000020,__READ       ,__irda_stat_bits);
__IO_REG32_BIT(IrDA_TFS,              0xD1000024,__WRITE      ,__irda_tfs_bits);
__IO_REG32_BIT(IrDA_RFS,              0xD1000028,__READ       ,__irda_rfs_bits);
__IO_REG32(    IrDA_TXB,              0xD100002C,__WRITE      );
__IO_REG32(    IrDA_RXB,              0xD1000030,__READ       );
__IO_REG32_BIT(IrDA_IMSC,             0xD10000E8,__READ_WRITE ,__irda_imsc_bits);
__IO_REG32_BIT(IrDA_RIS,              0xD10000EC,__READ       ,__irda_imsc_bits);
__IO_REG32_BIT(IrDA_MIS,              0xD10000F0,__READ       ,__irda_imsc_bits);
__IO_REG32_BIT(IrDA_ICR,              0xD10000F4,__WRITE      ,__irda_imsc_bits);
__IO_REG32_BIT(IrDA_ISR,              0xD10000F8,__WRITE      ,__irda_imsc_bits);
__IO_REG32_BIT(IrDA_DMA,              0xD10000FC,__READ_WRITE ,__irda_dma_bits);

/***************************************************************************
 **
 **  UART 0
 **
 ***************************************************************************/
__IO_REG16(    UART0DR,               0xD0000000,__READ_WRITE );
__IO_REG8_BIT( UART0RSR,              0xD0000004,__READ_WRITE ,__uartrsr_bits);
#define UART0ECR          UART0RSR
__IO_REG16_BIT(UART0FR,               0xD0000018,__READ       ,__uartfr_bits);
__IO_REG16(    UART0IBRD,             0xD0000024,__READ_WRITE );
__IO_REG8_BIT( UART0FBRD,             0xD0000028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG16_BIT(UART0LCR_H,            0xD000002C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG16_BIT(UART0CR,               0xD0000030,__READ_WRITE ,__uartcr_bits);
__IO_REG16_BIT(UART0IFLS,             0xD0000034,__READ_WRITE ,__uartifls_bits);
__IO_REG16_BIT(UART0IMSC,             0xD0000038,__READ_WRITE ,__uartimsc_bits);
__IO_REG16_BIT(UART0RIS,              0xD000003C,__READ       ,__uartris_bits);
__IO_REG16_BIT(UART0MIS,              0xD0000040,__READ       ,__uartmis_bits);
__IO_REG16_BIT(UART0ICR,              0xD0000044,__WRITE      ,__uarticr_bits);
__IO_REG16_BIT(UART0DMACR,            0xD0000048,__READ_WRITE ,__uartdmacr_bits);
__IO_REG32(    UART0PeriphID0,        0xD0000FE0,__READ       );
__IO_REG32(    UART0PeriphID1,        0xD0000FE4,__READ       );
__IO_REG32(    UART0PeriphID2,        0xD0000FE8,__READ       );
__IO_REG32(    UART0PeriphID3,        0xD0000FEC,__READ       );
__IO_REG32(    UART0PCellID0,         0xD0000FF0,__READ       );
__IO_REG32(    UART0PCellID1,         0xD0000FF4,__READ       );
__IO_REG32(    UART0PCellID2,         0xD0000FF8,__READ       );
__IO_REG32(    UART0PCellID3,         0xD0000FFC,__READ       );

/***************************************************************************
 **
 **  UART 1
 **
 ***************************************************************************/
__IO_REG16(    UART1DR,               0xB2000000,__READ_WRITE );
__IO_REG8_BIT( UART1RSR,              0xB2000004,__READ_WRITE ,__uartrsr_bits);
#define UART1ECR          UART1RSR
__IO_REG16_BIT(UART1FR,               0xB2000018,__READ       ,__uartfr_bits);
__IO_REG16(    UART1IBRD,             0xB2000024,__READ_WRITE );
__IO_REG8_BIT( UART1FBRD,             0xB2000028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG16_BIT(UART1LCR_H,            0xB200002C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG16_BIT(UART1CR,               0xB2000030,__READ_WRITE ,__uartcr_bits);
__IO_REG16_BIT(UART1IFLS,             0xB2000034,__READ_WRITE ,__uartifls_bits);
__IO_REG16_BIT(UART1IMSC,             0xB2000038,__READ_WRITE ,__uartimsc_bits);
__IO_REG16_BIT(UART1RIS,              0xB200003C,__READ       ,__uartris_bits);
__IO_REG16_BIT(UART1MIS,              0xB2000040,__READ       ,__uartmis_bits);
__IO_REG16_BIT(UART1ICR,              0xB2000044,__WRITE      ,__uarticr_bits);
__IO_REG16_BIT(UART1DMACR,            0xB2000048,__READ_WRITE ,__uartdmacr_bits);
__IO_REG32(    UART1PeriphID0,        0xB2000FE0,__READ       );
__IO_REG32(    UART1PeriphID1,        0xB2000FE4,__READ       );
__IO_REG32(    UART1PeriphID2,        0xB2000FE8,__READ       );
__IO_REG32(    UART1PeriphID3,        0xB2000FEC,__READ       );
__IO_REG32(    UART1PCellID0,         0xB2000FF0,__READ       );
__IO_REG32(    UART1PCellID1,         0xB2000FF4,__READ       );
__IO_REG32(    UART1PCellID2,         0xB2000FF8,__READ       );
__IO_REG32(    UART1PCellID3,         0xB2000FFC,__READ       );

/***************************************************************************
 **
 **  UART 2
 **
 ***************************************************************************/
__IO_REG16(    UART2DR,               0xB2080000,__READ_WRITE );
__IO_REG8_BIT( UART2RSR,              0xB2080004,__READ_WRITE ,__uartrsr_bits);
#define UART2ECR          UART2RSR
__IO_REG16_BIT(UART2FR,               0xB2080018,__READ       ,__uartfr_bits);
__IO_REG16(    UART2IBRD,             0xB2080024,__READ_WRITE );
__IO_REG8_BIT( UART2FBRD,             0xB2080028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG16_BIT(UART2LCR_H,            0xB208002C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG16_BIT(UART2CR,               0xB2080030,__READ_WRITE ,__uartcr_bits);
__IO_REG16_BIT(UART2IFLS,             0xB2080034,__READ_WRITE ,__uartifls_bits);
__IO_REG16_BIT(UART2IMSC,             0xB2080038,__READ_WRITE ,__uartimsc_bits);
__IO_REG16_BIT(UART2RIS,              0xB208003C,__READ       ,__uartris_bits);
__IO_REG16_BIT(UART2MIS,              0xB2080040,__READ       ,__uartmis_bits);
__IO_REG16_BIT(UART2ICR,              0xB2080044,__WRITE      ,__uarticr_bits);
__IO_REG16_BIT(UART2DMACR,            0xB2080048,__READ_WRITE ,__uartdmacr_bits);
__IO_REG32(    UART2PeriphID0,        0xB2080FE0,__READ       );
__IO_REG32(    UART2PeriphID1,        0xB2080FE4,__READ       );
__IO_REG32(    UART2PeriphID2,        0xB2080FE8,__READ       );
__IO_REG32(    UART2PeriphID3,        0xB2080FEC,__READ       );
__IO_REG32(    UART2PCellID0,         0xB2080FF0,__READ       );
__IO_REG32(    UART2PCellID1,         0xB2080FF4,__READ       );
__IO_REG32(    UART2PCellID2,         0xB2080FF8,__READ       );
__IO_REG32(    UART2PCellID3,         0xB2080FFC,__READ       );

/***************************************************************************
 **
 **  UART 3
 **
 ***************************************************************************/
__IO_REG16(    UART3DR,               0xB2100000,__READ_WRITE );
__IO_REG8_BIT( UART3RSR,              0xB2100004,__READ_WRITE ,__uartrsr_bits);
#define UART3ECR          UART3RSR
__IO_REG16_BIT(UART3FR,               0xB2100018,__READ       ,__uartfr_bits);
__IO_REG16(    UART3IBRD,             0xB2100024,__READ_WRITE );
__IO_REG8_BIT( UART3FBRD,             0xB2100028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG16_BIT(UART3LCR_H,            0xB210002C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG16_BIT(UART3CR,               0xB2100030,__READ_WRITE ,__uartcr_bits);
__IO_REG16_BIT(UART3IFLS,             0xB2100034,__READ_WRITE ,__uartifls_bits);
__IO_REG16_BIT(UART3IMSC,             0xB2100038,__READ_WRITE ,__uartimsc_bits);
__IO_REG16_BIT(UART3RIS,              0xB210003C,__READ       ,__uartris_bits);
__IO_REG16_BIT(UART3MIS,              0xB2100040,__READ       ,__uartmis_bits);
__IO_REG16_BIT(UART3ICR,              0xB2100044,__WRITE      ,__uarticr_bits);
__IO_REG16_BIT(UART3DMACR,            0xB2100048,__READ_WRITE ,__uartdmacr_bits);
__IO_REG32(    UART3PeriphID0,        0xB2100FE0,__READ       );
__IO_REG32(    UART3PeriphID1,        0xB2100FE4,__READ       );
__IO_REG32(    UART3PeriphID2,        0xB2100FE8,__READ       );
__IO_REG32(    UART3PeriphID3,        0xB2100FEC,__READ       );
__IO_REG32(    UART3PCellID0,         0xB2100FF0,__READ       );
__IO_REG32(    UART3PCellID1,         0xB2100FF4,__READ       );
__IO_REG32(    UART3PCellID2,         0xB2100FF8,__READ       );
__IO_REG32(    UART3PCellID3,         0xB2100FFC,__READ       );

/***************************************************************************
 **
 **  UART 4
 **
 ***************************************************************************/
__IO_REG16(    UART4DR,               0xB2180000,__READ_WRITE );
__IO_REG8_BIT( UART4RSR,              0xB2180004,__READ_WRITE ,__uartrsr_bits);
#define UART4ECR          UART4RSR
__IO_REG16_BIT(UART4FR,               0xB2180018,__READ       ,__uartfr_bits);
__IO_REG16(    UART4IBRD,             0xB2180024,__READ_WRITE );
__IO_REG8_BIT( UART4FBRD,             0xB2180028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG16_BIT(UART4LCR_H,            0xB218002C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG16_BIT(UART4CR,               0xB2180030,__READ_WRITE ,__uartcr_bits);
__IO_REG16_BIT(UART4IFLS,             0xB2180034,__READ_WRITE ,__uartifls_bits);
__IO_REG16_BIT(UART4IMSC,             0xB2180038,__READ_WRITE ,__uartimsc_bits);
__IO_REG16_BIT(UART4RIS,              0xB218003C,__READ       ,__uartris_bits);
__IO_REG16_BIT(UART4MIS,              0xB2180040,__READ       ,__uartmis_bits);
__IO_REG16_BIT(UART4ICR,              0xB2180044,__WRITE      ,__uarticr_bits);
__IO_REG16_BIT(UART4DMACR,            0xB2180048,__READ_WRITE ,__uartdmacr_bits);
__IO_REG32(    UART4PeriphID0,        0xB2180FE0,__READ       );
__IO_REG32(    UART4PeriphID1,        0xB2180FE4,__READ       );
__IO_REG32(    UART4PeriphID2,        0xB2180FE8,__READ       );
__IO_REG32(    UART4PeriphID3,        0xB2180FEC,__READ       );
__IO_REG32(    UART4PCellID0,         0xB2180FF0,__READ       );
__IO_REG32(    UART4PCellID1,         0xB2180FF4,__READ       );
__IO_REG32(    UART4PCellID2,         0xB2180FF8,__READ       );
__IO_REG32(    UART4PCellID3,         0xB2180FFC,__READ       );

/***************************************************************************
 **
 **  UART 5
 **
 ***************************************************************************/
__IO_REG16(    UART5DR,               0xB2200000,__READ_WRITE );
__IO_REG8_BIT( UART5RSR,              0xB2200004,__READ_WRITE ,__uartrsr_bits);
#define UART5ECR          UART5RSR
__IO_REG16_BIT(UART5FR,               0xB2200018,__READ       ,__uartfr_bits);
__IO_REG16(    UART5IBRD,             0xB2200024,__READ_WRITE );
__IO_REG8_BIT( UART5FBRD,             0xB2200028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG16_BIT(UART5LCR_H,            0xB220002C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG16_BIT(UART5CR,               0xB2200030,__READ_WRITE ,__uartcr_bits);
__IO_REG16_BIT(UART5IFLS,             0xB2200034,__READ_WRITE ,__uartifls_bits);
__IO_REG16_BIT(UART5IMSC,             0xB2200038,__READ_WRITE ,__uartimsc_bits);
__IO_REG16_BIT(UART5RIS,              0xB220003C,__READ       ,__uartris_bits);
__IO_REG16_BIT(UART5MIS,              0xB2200040,__READ       ,__uartmis_bits);
__IO_REG16_BIT(UART5ICR,              0xB2200044,__WRITE      ,__uarticr_bits);
__IO_REG16_BIT(UART5DMACR,            0xB2200048,__READ_WRITE ,__uartdmacr_bits);
__IO_REG32(    UART5PeriphID0,        0xB2200FE0,__READ       );
__IO_REG32(    UART5PeriphID1,        0xB2200FE4,__READ       );
__IO_REG32(    UART5PeriphID2,        0xB2200FE8,__READ       );
__IO_REG32(    UART5PeriphID3,        0xB2200FEC,__READ       );
__IO_REG32(    UART5PCellID0,         0xB2200FF0,__READ       );
__IO_REG32(    UART5PCellID1,         0xB2200FF4,__READ       );
__IO_REG32(    UART5PCellID2,         0xB2200FF8,__READ       );
__IO_REG32(    UART5PCellID3,         0xB2200FFC,__READ       );

/***************************************************************************
 **
 **  I2C
 **
 ***************************************************************************/
__IO_REG16_BIT(IC_CON,                0xD0180000,__READ_WRITE ,__ic_con_bits);
__IO_REG16_BIT(IC_TAR,                0xD0180004,__READ_WRITE ,__ic_tar_bits);
__IO_REG16_BIT(IC_SAR,                0xD0180008,__READ_WRITE ,__ic_sar_bits);
__IO_REG16_BIT(IC_HS_MADDR,           0xD018000C,__READ_WRITE ,__ic_hs_maddr_bits);
__IO_REG16_BIT(IC_DATA_CMD,           0xD0180010,__READ_WRITE ,__ic_data_cmd_bits);
__IO_REG16(    IC_SS_SCL_HCNT,        0xD0180014,__READ_WRITE );
__IO_REG16(    IC_SS_SCL_LCNT,        0xD0180018,__READ_WRITE );
__IO_REG16(    IC_FS_SCL_HCNT,        0xD018001C,__READ_WRITE );
__IO_REG16(    IC_FS_SCL_LCNT,        0xD0180020,__READ_WRITE );
__IO_REG16(    IC_HS_SCL_HCNT,        0xD0180024,__READ_WRITE );
__IO_REG16(    IC_HS_SCL_LCNT,        0xD0180028,__READ_WRITE );
__IO_REG16_BIT(IC_INTR_STAT,          0xD018002C,__READ       ,__ic_intr_stat_bits);
__IO_REG16_BIT(IC_INTR_MASK,          0xD0180030,__READ_WRITE ,__ic_intr_mask_bits);
__IO_REG16_BIT(IC_RAW_INTR_STAT,      0xD0180034,__READ       ,__ic_raw_intr_stat_bits);
__IO_REG16_BIT(IC_RX_TL,              0xD0180038,__READ_WRITE ,__ic_rx_tl_bits);
__IO_REG16_BIT(IC_TX_TL,              0xD018003C,__READ_WRITE ,__ic_tx_tl_bits);
__IO_REG16_BIT(IC_CLR_INTR,           0xD0180040,__READ       ,__ic_clr_intr_bits);
__IO_REG16(    IC_CLR_RX_UNDER,       0xD0180044,__READ       );
__IO_REG16(    IC_CLR_RX_OVER,        0xD0180048,__READ       );
__IO_REG16(    IC_CLR_TX_OVER,        0xD018004C,__READ       );
__IO_REG16(    IC_CLR_RD_REQ,         0xD0180050,__READ       );
__IO_REG16(    IC_CLR_TX_ABRT,        0xD0180054,__READ       );
__IO_REG16(    IC_CLR_RX_DONE,        0xD0180058,__READ       );
__IO_REG16(    IC_CLR_ACTIVITY,       0xD018005C,__READ       );
__IO_REG16(    IC_CLR_STOP_DET,       0xD0180060,__READ       );
__IO_REG16(    IC_CLR_START_DET,      0xD0180064,__READ       );
__IO_REG16(    IC_CLR_GEN_CALL,       0xD0180068,__READ       );
__IO_REG16_BIT(IC_ENABLE,             0xD018006C,__READ_WRITE ,__ic_enable_bits);
__IO_REG16_BIT(IC_STATUS,             0xD0180070,__READ       ,__ic_status_bits);
__IO_REG16_BIT(IC_TXFLR,              0xD0180074,__READ       ,__ic_txflr_bits);
__IO_REG16_BIT(IC_RXFLR,              0xD0180078,__READ       ,__ic_rxflr_bits);
__IO_REG16_BIT(IC_TX_ABRT_SOURCE,     0xD0180080,__READ_WRITE ,__ic_tx_abrt_source_bits);
__IO_REG16_BIT(IC_DMA_CR,             0xD0180088,__READ_WRITE ,__ic_dma_cr_bits);
__IO_REG16_BIT(IC_DMA_TDLR,           0xD018008C,__READ_WRITE ,__ic_dma_tdlr_bits);
__IO_REG16_BIT(IC_DMA_RDLR,           0xD0180090,__READ_WRITE ,__ic_dma_rdlr_bits);
__IO_REG32_BIT(IC_COMP_PARAM_1,       0xD01800F4,__READ       ,__ic_comp_param_1_bits);
__IO_REG32(    IC_COMP_VERSION,       0xD01800F8,__READ       );
__IO_REG32(    IC_COMP_TYPE,          0xD01800FC,__READ       );

/***************************************************************************
 **
 **  ADC
 **
 ***************************************************************************/
__IO_REG16_BIT(ADC_STATUS_REG,        0xD0080000,__READ_WRITE ,__adc_status_reg_bits);
__IO_REG16_BIT(ADC_AVERAGE_REG,       0xD0080004,__READ       ,__adc_average_reg_bits);
__IO_REG32(    ADC_SCAN_RATE_REG,     0xD0080008,__READ_WRITE );
__IO_REG16_BIT(ADC_CLK_REG,           0xD008000C,__READ_WRITE ,__adc_clk_reg_bits);
__IO_REG16_BIT(ADC_CH0_CTRL_REG,      0xD0080010,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH1_CTRL_REG,      0xD0080014,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH2_CTRL_REG,      0xD0080018,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH3_CTRL_REG,      0xD008001C,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH4_CTRL_REG,      0xD0080020,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH5_CTRL_REG,      0xD0080024,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH6_CTRL_REG,      0xD0080028,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH7_CTRL_REG,      0xD008002C,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG32_BIT(ADC_CH0_DATA_REG,      0xD0080030,__READ       ,__adc_ch_data_reg_bits);
__IO_REG32_BIT(ADC_CH1_DATA_REG,      0xD0080034,__READ       ,__adc_ch_data_reg_bits);
__IO_REG32_BIT(ADC_CH2_DATA_REG,      0xD0080038,__READ       ,__adc_ch_data_reg_bits);
__IO_REG32_BIT(ADC_CH3_DATA_REG,      0xD008003C,__READ       ,__adc_ch_data_reg_bits);
__IO_REG32_BIT(ADC_CH4_DATA_REG,      0xD0080040,__READ       ,__adc_ch_data_reg_bits);
__IO_REG32_BIT(ADC_CH5_DATA_REG,      0xD0080044,__READ       ,__adc_ch_data_reg_bits);
__IO_REG32_BIT(ADC_CH6_DATA_REG,      0xD0080048,__READ       ,__adc_ch_data_reg_bits);
__IO_REG32_BIT(ADC_CH7_DATA_REG,      0xD008004C,__READ       ,__adc_ch_data_reg_bits);

/***************************************************************************
 **
 **  RAS
 **
 ***************************************************************************/
__IO_REG32_BIT(RAS_BR,                0xB4000000,__READ       ,__ras_br_bits);
__IO_REG32_BIT(RAS_ISR,               0xB4000004,__READ       ,__ras_isr_bits);
__IO_REG32_BIT(RAS_FER,               0xB4000008,__READ_WRITE ,__ras_fer_bits);
__IO_REG32_BIT(RAS_SMIICFG,           0xB400000C,__READ_WRITE ,__ras_smiicfg_bits);
__IO_REG32_BIT(RAS_PL_GPIO_En0,       0xB4000010,__READ_WRITE ,__ras_pl_gpio_en0_bits);
__IO_REG32_BIT(RAS_PL_GPIO_En1,       0xB4000014,__READ_WRITE ,__ras_pl_gpio_en1_bits);
__IO_REG32_BIT(RAS_PL_GPIO_En2,       0xB4000018,__READ_WRITE ,__ras_pl_gpio_en2_bits);
__IO_REG32_BIT(RAS_PL_GPIO_En3,       0xB400001C,__READ_WRITE ,__ras_pl_gpio_en3_bits);
__IO_REG32_BIT(RAS_GPIO_DataOut0,     0xB4000020,__READ_WRITE ,__ras_gpio_dataout0_bits);
__IO_REG32_BIT(RAS_GPIO_DataOut1,     0xB4000024,__READ_WRITE ,__ras_gpio_dataout1_bits);
__IO_REG32_BIT(RAS_GPIO_DataOut2,     0xB4000028,__READ_WRITE ,__ras_gpio_dataout2_bits);
__IO_REG32_BIT(RAS_GPIO_Dir0,         0xB4000030,__READ_WRITE ,__ras_gpio_dataout0_bits);
__IO_REG32_BIT(RAS_GPIO_Dir1,         0xB4000034,__READ_WRITE ,__ras_gpio_dataout1_bits);
__IO_REG32_BIT(RAS_GPIO_Dir2,         0xB4000038,__READ_WRITE ,__ras_gpio_dataout2_bits);
__IO_REG32_BIT(RAS_GPIO_IntEn0,       0xB4000040,__READ_WRITE ,__ras_gpio_dataout0_bits);
__IO_REG32_BIT(RAS_GPIO_IntEn1,       0xB4000044,__READ_WRITE ,__ras_gpio_dataout1_bits);
__IO_REG32_BIT(RAS_GPIO_IntEn2,       0xB4000048,__READ_WRITE ,__ras_gpio_dataout2_bits);
__IO_REG32_BIT(RAS_GPIO_DataIn0,      0xB4000050,__READ       ,__ras_gpio_dataout0_bits);
__IO_REG32_BIT(RAS_GPIO_DataIn1,      0xB4000054,__READ       ,__ras_gpio_dataout1_bits);
__IO_REG32_BIT(RAS_GPIO_DataIn2,      0xB4000058,__READ       ,__ras_gpio_dataout2_bits);
__IO_REG32_BIT(RAS_GPIO_IntSts0,      0xB4000060,__READ       ,__ras_gpio_dataout0_bits);
__IO_REG32_BIT(RAS_GPIO_IntSts1,      0xB4000064,__READ       ,__ras_gpio_dataout1_bits);
__IO_REG32_BIT(RAS_GPIO_IntSts2,      0xB4000068,__READ       ,__ras_gpio_dataout2_bits);

/***************************************************************************
 **
 **  SMII MACB0
 **
 ***************************************************************************/
__IO_REG32_BIT(MACB0_NCTRL,           0xB0000000,__READ_WRITE ,__macb_nctrl_bits);
__IO_REG32_BIT(MACB0_NCFG,            0xB0000004,__READ_WRITE ,__macb_ncfg_bits);
__IO_REG32_BIT(MACB0_NST,             0xB0000008,__READ       ,__macb_nst_bits);
__IO_REG32_BIT(MACB0_TST,             0xB0000014,__READ_WRITE ,__macb_tst_bits);
__IO_REG32(    MACB0_RBQP,            0xB0000018,__READ_WRITE );
__IO_REG32(    MACB0_TBQP,            0xB000001C,__READ_WRITE );
__IO_REG32_BIT(MACB0_RST,             0xB0000020,__READ_WRITE ,__macb_rst_bits);
__IO_REG32_BIT(MACB0_IS,              0xB0000024,__READ_WRITE ,__macb_is_bits);
__IO_REG32_BIT(MACB0_IE,              0xB0000028,__WRITE      ,__macb_is_bits);
__IO_REG32_BIT(MACB0_ID,              0xB000002C,__WRITE      ,__macb_is_bits);
__IO_REG32_BIT(MACB0_IM,              0xB0000030,__READ       ,__macb_is_bits);
__IO_REG32_BIT(MACB0_PHYM,            0xB0000034,__READ_WRITE ,__macb_phym_bits);
__IO_REG32_BIT(MACB0_PT,              0xB0000038,__READ       ,__macb_pt_bits);
__IO_REG32_BIT(MACB0_PFR,             0xB000003C,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB0_FT,              0xB0000040,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB0_SCF,             0xB0000044,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB0_MCF,             0xB0000048,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB0_FR,              0xB000004C,__READ_WRITE ,__macb_fr_bits);
__IO_REG32_BIT(MACB0_FCSE,            0xB0000050,__READ_WRITE ,__macb_fcse_bits);
__IO_REG32_BIT(MACB0_AE,              0xB0000054,__READ_WRITE ,__macb_ae_bits);
__IO_REG32_BIT(MACB0_DTF,             0xB0000058,__READ_WRITE ,__macb_dtf_bits);
__IO_REG32_BIT(MACB0_LC,              0xB000005C,__READ_WRITE ,__macb_lc_bits);
__IO_REG32_BIT(MACB0_EC,              0xB0000060,__READ_WRITE ,__macb_ec_bits);
__IO_REG32_BIT(MACB0_TUE,             0xB0000064,__READ_WRITE ,__macb_tue_bits);
__IO_REG32_BIT(MACB0_CSE,             0xB0000068,__READ_WRITE ,__macb_cse_bits);
__IO_REG32_BIT(MACB0_RRE,             0xB000006C,__READ_WRITE ,__macb_rre_bits);
__IO_REG32_BIT(MACB0_ROE,             0xB0000070,__READ_WRITE ,__macb_roe_bits);
__IO_REG32_BIT(MACB0_RSE,             0xB0000074,__READ_WRITE ,__macb_rse_bits);
__IO_REG32_BIT(MACB0_ELE,             0xB0000078,__READ_WRITE ,__macb_ele_bits);
__IO_REG32_BIT(MACB0_RJ,              0xB000007C,__READ_WRITE ,__macb_rj_bits);
__IO_REG32_BIT(MACB0_UF,              0xB0000080,__READ_WRITE ,__macb_uf_bits);
__IO_REG32_BIT(MACB0_SEQTE,           0xB0000084,__READ_WRITE ,__macb_seqte_bits);
__IO_REG32_BIT(MACB0_RLFM,            0xB0000088,__READ_WRITE ,__macb_rlfm_bits);
__IO_REG32_BIT(MACB0_TPF,             0xB000008C,__READ_WRITE ,__macb_tpf_bits);
__IO_REG32(    MACB0_HRB,             0xB0000090,__READ_WRITE );
__IO_REG32(    MACB0_HRT,             0xB0000094,__READ_WRITE );
__IO_REG32(    MACB0_SAB1,            0xB0000098,__READ_WRITE );
__IO_REG32_BIT(MACB0_SAT1,            0xB000009C,__READ_WRITE ,__macb_sat1_bits);
__IO_REG32(    MACB0_SAB2,            0xB00000A0,__READ_WRITE );
__IO_REG32_BIT(MACB0_SAT2,            0xB00000A4,__READ_WRITE ,__macb_sat2_bits);
__IO_REG32(    MACB0_SAB3,            0xB00000A8,__READ_WRITE );
__IO_REG32_BIT(MACB0_SAT3,            0xB00000AC,__READ_WRITE ,__macb_sat3_bits);
__IO_REG32(    MACB0_SAB4,            0xB00000B0,__READ_WRITE );
__IO_REG32_BIT(MACB0_SAT4,            0xB00000B4,__READ_WRITE ,__macb_sat4_bits);
__IO_REG32_BIT(MACB0_TIDC,            0xB00000B8,__READ_WRITE ,__macb_tidc_bits);
__IO_REG32_BIT(MACB0_TPQ,             0xB00000BC,__READ_WRITE ,__macb_tpq_bits);
__IO_REG32_BIT(MACB0_WOLAN,           0xB00000C4,__READ_WRITE ,__macb_wolan_bits);
__IO_REG32_BIT(MACB0_RR,              0xB00000FC,__READ_WRITE ,__macb_rr_bits);

/***************************************************************************
 **
 **  SMII MACB1
 **
 ***************************************************************************/
__IO_REG32_BIT(MACB1_NCTRL,           0xB0800000,__READ_WRITE ,__macb_nctrl_bits);
__IO_REG32_BIT(MACB1_NCFG,            0xB0800004,__READ_WRITE ,__macb_ncfg_bits);
__IO_REG32_BIT(MACB1_NST,             0xB0800008,__READ       ,__macb_nst_bits);
__IO_REG32_BIT(MACB1_TST,             0xB0800014,__READ_WRITE ,__macb_tst_bits);
__IO_REG32(    MACB1_RBQP,            0xB0800018,__READ_WRITE );
__IO_REG32(    MACB1_TBQP,            0xB080001C,__READ_WRITE );
__IO_REG32_BIT(MACB1_RST,             0xB0800020,__READ_WRITE ,__macb_rst_bits);
__IO_REG32_BIT(MACB1_IS,              0xB0800024,__READ_WRITE ,__macb_is_bits);
__IO_REG32_BIT(MACB1_IE,              0xB0800028,__WRITE      ,__macb_is_bits);
__IO_REG32_BIT(MACB1_ID,              0xB080002C,__WRITE      ,__macb_is_bits);
__IO_REG32_BIT(MACB1_IM,              0xB0800030,__READ       ,__macb_is_bits);
__IO_REG32_BIT(MACB1_PHYM,            0xB0800034,__READ_WRITE ,__macb_phym_bits);
__IO_REG32_BIT(MACB1_PT,              0xB0800038,__READ       ,__macb_pt_bits);
__IO_REG32_BIT(MACB1_PFR,             0xB080003C,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB1_FT,              0xB0800040,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB1_SCF,             0xB0800044,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB1_MCF,             0xB0800048,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB1_FR,              0xB080004C,__READ_WRITE ,__macb_fr_bits);
__IO_REG32_BIT(MACB1_FCSE,            0xB0800050,__READ_WRITE ,__macb_fcse_bits);
__IO_REG32_BIT(MACB1_AE,              0xB0800054,__READ_WRITE ,__macb_ae_bits);
__IO_REG32_BIT(MACB1_DTF,             0xB0800058,__READ_WRITE ,__macb_dtf_bits);
__IO_REG32_BIT(MACB1_LC,              0xB080005C,__READ_WRITE ,__macb_lc_bits);
__IO_REG32_BIT(MACB1_EC,              0xB0800060,__READ_WRITE ,__macb_ec_bits);
__IO_REG32_BIT(MACB1_TUE,             0xB0800064,__READ_WRITE ,__macb_tue_bits);
__IO_REG32_BIT(MACB1_CSE,             0xB0800068,__READ_WRITE ,__macb_cse_bits);
__IO_REG32_BIT(MACB1_RRE,             0xB080006C,__READ_WRITE ,__macb_rre_bits);
__IO_REG32_BIT(MACB1_ROE,             0xB0800070,__READ_WRITE ,__macb_roe_bits);
__IO_REG32_BIT(MACB1_RSE,             0xB0800074,__READ_WRITE ,__macb_rse_bits);
__IO_REG32_BIT(MACB1_ELE,             0xB0800078,__READ_WRITE ,__macb_ele_bits);
__IO_REG32_BIT(MACB1_RJ,              0xB080007C,__READ_WRITE ,__macb_rj_bits);
__IO_REG32_BIT(MACB1_UF,              0xB0800080,__READ_WRITE ,__macb_uf_bits);
__IO_REG32_BIT(MACB1_SEQTE,           0xB0800084,__READ_WRITE ,__macb_seqte_bits);
__IO_REG32_BIT(MACB1_RLFM,            0xB0800088,__READ_WRITE ,__macb_rlfm_bits);
__IO_REG32_BIT(MACB1_TPF,             0xB080008C,__READ_WRITE ,__macb_tpf_bits);
__IO_REG32(    MACB1_HRB,             0xB0800090,__READ_WRITE );
__IO_REG32(    MACB1_HRT,             0xB0800094,__READ_WRITE );
__IO_REG32(    MACB1_SAB1,            0xB0800098,__READ_WRITE );
__IO_REG32_BIT(MACB1_SAT1,            0xB080009C,__READ_WRITE ,__macb_sat1_bits);
__IO_REG32(    MACB1_SAB2,            0xB08000A0,__READ_WRITE );
__IO_REG32_BIT(MACB1_SAT2,            0xB08000A4,__READ_WRITE ,__macb_sat2_bits);
__IO_REG32(    MACB1_SAB3,            0xB08000A8,__READ_WRITE );
__IO_REG32_BIT(MACB1_SAT3,            0xB08000AC,__READ_WRITE ,__macb_sat3_bits);
__IO_REG32(    MACB1_SAB4,            0xB08000B0,__READ_WRITE );
__IO_REG32_BIT(MACB1_SAT4,            0xB08000B4,__READ_WRITE ,__macb_sat4_bits);
__IO_REG32_BIT(MACB1_TIDC,            0xB08000B8,__READ_WRITE ,__macb_tidc_bits);
__IO_REG32_BIT(MACB1_TPQ,             0xB08000BC,__READ_WRITE ,__macb_tpq_bits);
__IO_REG32_BIT(MACB1_WOLAN,           0xB08000C4,__READ_WRITE ,__macb_wolan_bits);
__IO_REG32_BIT(MACB1_RR,              0xB08000FC,__READ_WRITE ,__macb_rr_bits);

/***************************************************************************
 **
 **  SMII MACB2
 **
 ***************************************************************************/
__IO_REG32_BIT(MACB2_NCTRL,           0xB1000000,__READ_WRITE ,__macb_nctrl_bits);
__IO_REG32_BIT(MACB2_NCFG,            0xB1000004,__READ_WRITE ,__macb_ncfg_bits);
__IO_REG32_BIT(MACB2_NST,             0xB1000008,__READ       ,__macb_nst_bits);
__IO_REG32_BIT(MACB2_TST,             0xB1000014,__READ_WRITE ,__macb_tst_bits);
__IO_REG32(    MACB2_RBQP,            0xB1000018,__READ_WRITE );
__IO_REG32(    MACB2_TBQP,            0xB100001C,__READ_WRITE );
__IO_REG32_BIT(MACB2_RST,             0xB1000020,__READ_WRITE ,__macb_rst_bits);
__IO_REG32_BIT(MACB2_IS,              0xB1000024,__READ_WRITE ,__macb_is_bits);
__IO_REG32_BIT(MACB2_IE,              0xB1000028,__WRITE      ,__macb_is_bits);
__IO_REG32_BIT(MACB2_ID,              0xB100002C,__WRITE      ,__macb_is_bits);
__IO_REG32_BIT(MACB2_IM,              0xB1000030,__READ       ,__macb_is_bits);
__IO_REG32_BIT(MACB2_PHYM,            0xB1000034,__READ_WRITE ,__macb_phym_bits);
__IO_REG32_BIT(MACB2_PT,              0xB1000038,__READ       ,__macb_pt_bits);
__IO_REG32_BIT(MACB2_PFR,             0xB100003C,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB2_FT,              0xB1000040,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB2_SCF,             0xB1000044,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB2_MCF,             0xB1000048,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB2_FR,              0xB100004C,__READ_WRITE ,__macb_fr_bits);
__IO_REG32_BIT(MACB2_FCSE,            0xB1000050,__READ_WRITE ,__macb_fcse_bits);
__IO_REG32_BIT(MACB2_AE,              0xB1000054,__READ_WRITE ,__macb_ae_bits);
__IO_REG32_BIT(MACB2_DTF,             0xB1000058,__READ_WRITE ,__macb_dtf_bits);
__IO_REG32_BIT(MACB2_LC,              0xB100005C,__READ_WRITE ,__macb_lc_bits);
__IO_REG32_BIT(MACB2_EC,              0xB1000060,__READ_WRITE ,__macb_ec_bits);
__IO_REG32_BIT(MACB2_TUE,             0xB1000064,__READ_WRITE ,__macb_tue_bits);
__IO_REG32_BIT(MACB2_CSE,             0xB1000068,__READ_WRITE ,__macb_cse_bits);
__IO_REG32_BIT(MACB2_RRE,             0xB100006C,__READ_WRITE ,__macb_rre_bits);
__IO_REG32_BIT(MACB2_ROE,             0xB1000070,__READ_WRITE ,__macb_roe_bits);
__IO_REG32_BIT(MACB2_RSE,             0xB1000074,__READ_WRITE ,__macb_rse_bits);
__IO_REG32_BIT(MACB2_ELE,             0xB1000078,__READ_WRITE ,__macb_ele_bits);
__IO_REG32_BIT(MACB2_RJ,              0xB100007C,__READ_WRITE ,__macb_rj_bits);
__IO_REG32_BIT(MACB2_UF,              0xB1000080,__READ_WRITE ,__macb_uf_bits);
__IO_REG32_BIT(MACB2_SEQTE,           0xB1000084,__READ_WRITE ,__macb_seqte_bits);
__IO_REG32_BIT(MACB2_RLFM,            0xB1000088,__READ_WRITE ,__macb_rlfm_bits);
__IO_REG32_BIT(MACB2_TPF,             0xB100008C,__READ_WRITE ,__macb_tpf_bits);
__IO_REG32(    MACB2_HRB,             0xB1000090,__READ_WRITE );
__IO_REG32(    MACB2_HRT,             0xB1000094,__READ_WRITE );
__IO_REG32(    MACB2_SAB1,            0xB1000098,__READ_WRITE );
__IO_REG32_BIT(MACB2_SAT1,            0xB100009C,__READ_WRITE ,__macb_sat1_bits);
__IO_REG32(    MACB2_SAB2,            0xB10000A0,__READ_WRITE );
__IO_REG32_BIT(MACB2_SAT2,            0xB10000A4,__READ_WRITE ,__macb_sat2_bits);
__IO_REG32(    MACB2_SAB3,            0xB10000A8,__READ_WRITE );
__IO_REG32_BIT(MACB2_SAT3,            0xB10000AC,__READ_WRITE ,__macb_sat3_bits);
__IO_REG32(    MACB2_SAB4,            0xB10000B0,__READ_WRITE );
__IO_REG32_BIT(MACB2_SAT4,            0xB10000B4,__READ_WRITE ,__macb_sat4_bits);
__IO_REG32_BIT(MACB2_TIDC,            0xB10000B8,__READ_WRITE ,__macb_tidc_bits);
__IO_REG32_BIT(MACB2_TPQ,             0xB10000BC,__READ_WRITE ,__macb_tpq_bits);
__IO_REG32_BIT(MACB2_WOLAN,           0xB10000C4,__READ_WRITE ,__macb_wolan_bits);
__IO_REG32_BIT(MACB2_RR,              0xB10000FC,__READ_WRITE ,__macb_rr_bits);

/***************************************************************************
 **
 **  SMII MACB3
 **
 ***************************************************************************/
__IO_REG32_BIT(MACB3_NCTRL,           0xB1800000,__READ_WRITE ,__macb_nctrl_bits);
__IO_REG32_BIT(MACB3_NCFG,            0xB1800004,__READ_WRITE ,__macb_ncfg_bits);
__IO_REG32_BIT(MACB3_NST,             0xB1800008,__READ       ,__macb_nst_bits);
__IO_REG32_BIT(MACB3_TST,             0xB1800014,__READ_WRITE ,__macb_tst_bits);
__IO_REG32(    MACB3_RBQP,            0xB1800018,__READ_WRITE );
__IO_REG32(    MACB3_TBQP,            0xB180001C,__READ_WRITE );
__IO_REG32_BIT(MACB3_RST,             0xB1800020,__READ_WRITE ,__macb_rst_bits);
__IO_REG32_BIT(MACB3_IS,              0xB1800024,__READ_WRITE ,__macb_is_bits);
__IO_REG32_BIT(MACB3_IE,              0xB1800028,__WRITE      ,__macb_is_bits);
__IO_REG32_BIT(MACB3_ID,              0xB180002C,__WRITE      ,__macb_is_bits);
__IO_REG32_BIT(MACB3_IM,              0xB1800030,__READ       ,__macb_is_bits);
__IO_REG32_BIT(MACB3_PHYM,            0xB1800034,__READ_WRITE ,__macb_phym_bits);
__IO_REG32_BIT(MACB3_PT,              0xB1800038,__READ       ,__macb_pt_bits);
__IO_REG32_BIT(MACB3_PFR,             0xB180003C,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB3_FT,              0xB1800040,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB3_SCF,             0xB1800044,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB3_MCF,             0xB1800048,__READ_WRITE ,__macb_pfr_bits);
__IO_REG32_BIT(MACB3_FR,              0xB180004C,__READ_WRITE ,__macb_fr_bits);
__IO_REG32_BIT(MACB3_FCSE,            0xB1800050,__READ_WRITE ,__macb_fcse_bits);
__IO_REG32_BIT(MACB3_AE,              0xB1800054,__READ_WRITE ,__macb_ae_bits);
__IO_REG32_BIT(MACB3_DTF,             0xB1800058,__READ_WRITE ,__macb_dtf_bits);
__IO_REG32_BIT(MACB3_LC,              0xB180005C,__READ_WRITE ,__macb_lc_bits);
__IO_REG32_BIT(MACB3_EC,              0xB1800060,__READ_WRITE ,__macb_ec_bits);
__IO_REG32_BIT(MACB3_TUE,             0xB1800064,__READ_WRITE ,__macb_tue_bits);
__IO_REG32_BIT(MACB3_CSE,             0xB1800068,__READ_WRITE ,__macb_cse_bits);
__IO_REG32_BIT(MACB3_RRE,             0xB180006C,__READ_WRITE ,__macb_rre_bits);
__IO_REG32_BIT(MACB3_ROE,             0xB1800070,__READ_WRITE ,__macb_roe_bits);
__IO_REG32_BIT(MACB3_RSE,             0xB1800074,__READ_WRITE ,__macb_rse_bits);
__IO_REG32_BIT(MACB3_ELE,             0xB1800078,__READ_WRITE ,__macb_ele_bits);
__IO_REG32_BIT(MACB3_RJ,              0xB180007C,__READ_WRITE ,__macb_rj_bits);
__IO_REG32_BIT(MACB3_UF,              0xB1800080,__READ_WRITE ,__macb_uf_bits);
__IO_REG32_BIT(MACB3_SEQTE,           0xB1800084,__READ_WRITE ,__macb_seqte_bits);
__IO_REG32_BIT(MACB3_RLFM,            0xB1800088,__READ_WRITE ,__macb_rlfm_bits);
__IO_REG32_BIT(MACB3_TPF,             0xB180008C,__READ_WRITE ,__macb_tpf_bits);
__IO_REG32(    MACB3_HRB,             0xB1800090,__READ_WRITE );
__IO_REG32(    MACB3_HRT,             0xB1800094,__READ_WRITE );
__IO_REG32(    MACB3_SAB1,            0xB1800098,__READ_WRITE );
__IO_REG32_BIT(MACB3_SAT1,            0xB180009C,__READ_WRITE ,__macb_sat1_bits);
__IO_REG32(    MACB3_SAB2,            0xB18000A0,__READ_WRITE );
__IO_REG32_BIT(MACB3_SAT2,            0xB18000A4,__READ_WRITE ,__macb_sat2_bits);
__IO_REG32(    MACB3_SAB3,            0xB18000A8,__READ_WRITE );
__IO_REG32_BIT(MACB3_SAT3,            0xB18000AC,__READ_WRITE ,__macb_sat3_bits);
__IO_REG32(    MACB3_SAB4,            0xB18000B0,__READ_WRITE );
__IO_REG32_BIT(MACB3_SAT4,            0xB18000B4,__READ_WRITE ,__macb_sat4_bits);
__IO_REG32_BIT(MACB3_TIDC,            0xB18000B8,__READ_WRITE ,__macb_tidc_bits);
__IO_REG32_BIT(MACB3_TPQ,             0xB18000BC,__READ_WRITE ,__macb_tpq_bits);
__IO_REG32_BIT(MACB3_WOLAN,           0xB18000C4,__READ_WRITE ,__macb_wolan_bits);
__IO_REG32_BIT(MACB3_RR,              0xB18000FC,__READ_WRITE ,__macb_rr_bits);

/***************************************************************************
 **
 **  FSMC
 **
 ***************************************************************************/
__IO_REG32_BIT(GenMemCtrl_PC0,        0x44000040,__READ_WRITE ,__genmemctrl_pc_bits);
__IO_REG32_BIT(GenMemCtrl_Comm0,      0x44000048,__READ_WRITE ,__genmemctrl_comm_bits);
__IO_REG32_BIT(GenMemCtrl_Attrib0,    0x4400004C,__READ_WRITE ,__genmemctrl_comm_bits);
__IO_REG32_BIT(GenMemCtrl_ECCr0,      0x44000054,__READ       ,__genmemctrl_eccr_bits);
__IO_REG32(    GenMemCtrl_PeriphID0,  0x44000FE0,__READ       );
__IO_REG32(    GenMemCtrl_PeriphID1,  0x44000FE4,__READ       );
__IO_REG32(    GenMemCtrl_PeriphID2,  0x44000FE8,__READ       );
__IO_REG32(    GenMemCtrl_PeriphID3,  0x44000FEC,__READ       );
__IO_REG32(    GenMemCtrl_PCellID0,   0x44000FF0,__READ       );
__IO_REG32(    GenMemCtrl_PCellID1,   0x44000FF4,__READ       );
__IO_REG32(    GenMemCtrl_PCellID2,   0x44000FF8,__READ       );
__IO_REG32(    GenMemCtrl_PCellID3,   0x44000FFC,__READ       );

/***************************************************************************
 **
 **  RS485 HDLC 1
 **
 ***************************************************************************/
__IO_REG32_BIT(HDLC1_HTCR,            0xB3000020,__READ_WRITE ,__hdlc_htcr_bits);
__IO_REG32_BIT(HDLC1_HRCR,            0xB3000030,__READ_WRITE ,__hdlc_hrcr_bits);
__IO_REG32_BIT(HDLC1_AFRAR,           0xB3000034,__READ_WRITE ,__hdlc_afrar_bits);
__IO_REG16(    HDLC1_AFRDR,           0xB3000038,__READ_WRITE );
__IO_REG32(    HDLC1_IBAR,            0xB3000040,__READ_WRITE );
__IO_REG32_BIT(HDLC1_IMR,             0xB3000044,__READ_WRITE ,__hdlc_imr_bits);
__IO_REG32_BIT(HDLC1_IR,              0xB3000048,__READ       ,__hdlc_ir_bits);
__IO_REG32_BIT(HDLC1_IQSR,            0xB300004C,__READ_WRITE ,__hdlc_iqsr_bits);

/***************************************************************************
 **
 **  RS485 HDLC 2
 **
 ***************************************************************************/
__IO_REG32_BIT(HDLC2_HTCR,            0xB3800020,__READ_WRITE ,__hdlc_htcr_bits);
__IO_REG32_BIT(HDLC2_HRCR,            0xB3800030,__READ_WRITE ,__hdlc_hrcr_bits);
__IO_REG32_BIT(HDLC2_AFRAR,           0xB3800034,__READ_WRITE ,__hdlc_afrar_bits);
__IO_REG16(    HDLC2_AFRDR,           0xB3800038,__READ_WRITE );
__IO_REG32(    HDLC2_IBAR,            0xB3800040,__READ_WRITE );
__IO_REG32_BIT(HDLC2_IMR,             0xB3800044,__READ_WRITE ,__hdlc_imr_bits);
__IO_REG32_BIT(HDLC2_IR,              0xB3800048,__READ       ,__hdlc_ir_bits);
__IO_REG32_BIT(HDLC2_IQSR,            0xB380004C,__READ_WRITE ,__hdlc_iqsr_bits);

/***************************************************************************
 **
 **  TDM/E1 HDLC
 **
 ***************************************************************************/
__IO_REG32_BIT(TDM_RX_TAAR,           0xB2800000,__READ_WRITE ,__tdm_rx_taar_bits);
__IO_REG32_BIT(TDM_RX_TADR,           0xB2800004,__READ_WRITE ,__tdm_rx_tadr_bits);
__IO_REG32_BIT(TDM_TX_TAAR,           0xB2800010,__READ_WRITE ,__tdm_rx_taar_bits);
__IO_REG32_BIT(TDM_TX_TADR,           0xB2800014,__READ_WRITE ,__tdm_rx_tadr_bits);
__IO_REG32_BIT(TDM_HTCR,              0xB2800020,__READ_WRITE ,__tdm_htcr_bits);
__IO_REG32_BIT(TDM_HRCR,              0xB2800030,__READ_WRITE ,__tdm_hrcr_bits);
__IO_REG32_BIT(TDM_AFRAR,             0xB2800034,__READ_WRITE ,__tdm_afrar_bits);
__IO_REG32(    TDM_AFRDR,             0xB2800038,__READ_WRITE );
__IO_REG32(    TDM_IBAR,              0xB2800040,__READ_WRITE );
__IO_REG32_BIT(TDM_IMR,               0xB2800044,__READ_WRITE ,__tdm_imr_bits);
__IO_REG32_BIT(TDM_IR,                0xB2800048,__READ       ,__tdm_ir_bits);
__IO_REG32_BIT(TDM_IQSR,              0xB280004C,__READ_WRITE ,__tdm_iqsr_bits);

/***************************************************************************
 **  Assembler specific declarations
 ***************************************************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  Interrupt vector table
 **
 ***************************************************************************/
#define RESETV  0x00  /* Reset                           */
#define UNDEFV  0x04  /* Undefined instruction           */
#define SWIV    0x08  /* Software interrupt              */
#define PABORTV 0x0c  /* Prefetch abort                  */
#define DABORTV 0x10  /* Data abort                      */
#define IRQV    0x18  /* Normal interrupt                */
#define FIQV    0x1c  /* Fast interrupt                  */

#endif    /* __IOSPEAR310_H */
