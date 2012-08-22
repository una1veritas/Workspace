//=========================================================
// LPC1114 Project
//=========================================================
// File Name : fixedpoint.h
// Function  : Fixed Point Library Header
//---------------------------------------------------------
// Rev.01 2010.08.17 Munetomo Maruyama
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

#ifndef __FIXEDPOINT_H__
#define __FIXEDPOINT_H__

#ifdef __USE_CMSIS
#include "LPC11xx.h"
#endif

//#include <stdint.h>

//==================
// Define Types
//==================
typedef signed long fix32_t;

//=====================
// Define Constants
//=====================
#define FIXQ 16
#define FIX_PI 0x0003243f // 3.14

//===================
// Define Operations
//===================
#define MIN(x, y) (((x) < (y))? x : y)
#define MAX(x, y) (((x) > (y))? x : y)

//==================
// Prototypes
//==================
fix32_t FIX_INT(signed long ia);
signed long INT_FIX(fix32_t fa);
fix32_t FIX_DBL(double da);
double  DBL_FIX(fix32_t fa);
//
fix32_t FIX_Add(fix32_t fa, fix32_t fb);
fix32_t FIX_Sub(fix32_t fa, fix32_t fb);
fix32_t FIX_Abs(fix32_t fa);
fix32_t FIX_Mul(fix32_t fa, fix32_t fb);
fix32_t FIX_Div(fix32_t fa, fix32_t fb);
fix32_t FIX_Sqrt(fix32_t fa);
//
fix32_t FIX_Sin(fix32_t frad);
fix32_t FIX_Cos(fix32_t frad);
fix32_t FIX_Atan(fix32_t fval);
fix32_t FIX_Atan2(fix32_t fnum, fix32_t fden);
fix32_t FIX_Deg_Rad(fix32_t frad);
fix32_t FIX_Rad_Deg(fix32_t fdeg);
fix32_t FIX_Sin_Deg(fix32_t fdeg);
fix32_t FIX_Cos_Deg(fix32_t fdeg);
fix32_t FIX_Atan_Deg(fix32_t fval);
fix32_t FIX_Atan2_Deg(fix32_t fnum, fix32_t fden);

#endif // __FIXEDPOINT_H__

//=========================================================
// End of Program
//=========================================================
