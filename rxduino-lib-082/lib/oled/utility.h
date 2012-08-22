//=========================================================
// LPC1114 Project
//=========================================================
// File Name : utiity.h
// Function  : Utility Routine Header
//---------------------------------------------------------
// Rev.01 2010.08.29 Munetomo Maruyama
//---------------------------------------------------------
// Copyright (C) 2010-2011 Munetomo Maruyama
//=========================================================
// ---- License Information -------------------------------
// Anyone can FREELY use this code fully or partially
// under conditions shown below.
// 1. You may use this code only for individual purpose,
//    and educational purpose.
//    Do not use this code for business even if partially.
// 2. You can copy, modify and distribute this code.
// 3. You should remain this header text in your codes
//   including Copyright credit and License Information.
// 4. Your codes should inherit this license information.
//=========================================================
// ---- Patent Notice -------------------------------------
// I have not cared whether this system (hw + sw) causes
// infringement on the patent, copyright, trademark,
// or trade secret rights of others. You have all
// responsibilities for determining if your designs
// and products infringe on the intellectual property
// rights of others, when you use technical information
// included in this system for your business.
//=========================================================
// ---- Disclaimers ---------------------------------------
// The function and reliability of this system are not
// guaranteed. They may cause any damages to loss of
// properties, data, money, profits, life, or business.
// By adopting this system even partially, you assume
// all responsibility for its use.
//=========================================================

#ifndef __UTILITY_H__
#define __UTILITY_H__

#ifdef __USE_CMSIS
#include "LPC11xx.h"
#endif

#include <stdarg.h>
#include "fixedpoint.h"

//=============
// Prototypes
//=============
signed long power(signed long x, signed long n);
unsigned char BCD_INT(unsigned char num);
unsigned char INT_BCD(unsigned char bcd);
unsigned long xatoi(unsigned char **str, signed long *data);
unsigned char *xitoa(unsigned char *str, unsigned long *pLength, signed long value, signed char radix, signed char width);
unsigned char *xsnprintf(unsigned char *str, unsigned long length, const char *format, ...);
unsigned char *xvsnprintf(unsigned char *str, unsigned long length, const char *format, va_list ap);

unsigned char* Append_String_UI32 (unsigned long value, unsigned char *string, unsigned long radix, unsigned long length);
unsigned char* Append_String_SI32 (signed long value, unsigned char *string, unsigned long radix, unsigned long length);
unsigned char* Append_String_Fixed(fix32_t fvalue, unsigned char *string, unsigned long format);

unsigned char* Append_String_UI08_Ndigit(unsigned long value, unsigned char *string, unsigned long digit);
unsigned char* Append_String_UI08_Ndigit(unsigned long value, unsigned char *string, unsigned long digit);
unsigned char* Append_String(unsigned char *pDst, unsigned char *pSrc);


#endif // __UTILITY_H__

//=========================================================
// End of Program
//=========================================================
