//=========================================================
// LPC1114 Project
//=========================================================
// File Name : oled.h
// Function  : OLED Control Header
//---------------------------------------------------------
// Rev.01 2010.08.13 Munetomo Maruyama
// Rev.02 2011.03.20 Munetomo Maruyama
//        - Add Orientation Mode
//        - Add OLED_GRY (Color Definition)
//        - Correct OLED_printf_Color()
//        - Correct OLED SPI Clock (24MHz-->12MHz)
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


#ifndef __OLED_H__
#define __OLED_H__

#ifdef __USE_CMSIS
#include "LPC11xx.h"
#endif

//===============
// Define Colors
//===============
#define OLED_RED_MAX 0x1f
#define OLED_GRN_MAX 0x3f
#define OLED_BLU_MAX 0x1f
#define OLED_RED_MIN 0x00
#define OLED_GRN_MIN 0x00
#define OLED_BLU_MIN 0x00
#define OLED_RED_MID 0x10
#define OLED_GRN_MID 0x20
#define OLED_BLU_MID 0x10
//
#define OLED_RED ((OLED_RED_MAX << 11) + (OLED_GRN_MIN << 5) + (OLED_BLU_MIN << 0))
#define OLED_GRN ((OLED_RED_MIN << 11) + (OLED_GRN_MAX << 5) + (OLED_BLU_MIN << 0))
#define OLED_BLU ((OLED_RED_MIN << 11) + (OLED_GRN_MIN << 5) + (OLED_BLU_MAX << 0))
#define OLED_BLK ((OLED_RED_MIN << 11) + (OLED_GRN_MIN << 5) + (OLED_BLU_MIN << 0))
#define OLED_WHT ((OLED_RED_MAX << 11) + (OLED_GRN_MAX << 5) + (OLED_BLU_MAX << 0))
#define OLED_YEL ((OLED_RED_MAX << 11) + (OLED_GRN_MAX << 5) + (OLED_BLU_MIN << 0))
#define OLED_CYN ((OLED_RED_MIN << 11) + (OLED_GRN_MAX << 5) + (OLED_BLU_MAX << 0))
#define OLED_MAG ((OLED_RED_MAX << 11) + (OLED_GRN_MIN << 5) + (OLED_BLU_MAX << 0))
#define OLED_GRY ((OLED_RED_MID << 11) + (OLED_GRN_MID << 5) + (OLED_BLU_MID << 0))

//=================
// Font Parameters
//=================
#define OLED_FONT_SMALL  0
#define OLED_FONT_MEDIUM 1
#define OLED_FONT_LARGE  2

//=================
// Orientation Mode
//=================
enum OELD_ORIENTATION_MODE {OLED_TOP_N, OLED_TOP_W, OLED_TOP_S, OLED_TOP_E};

//==============
// Prototypes
//==============
int Init_OLED(char portnum);
void Init_OLED_with_Orientation(char portnum,unsigned long mode);
void OLED_Send_Command(char portnum,unsigned long *oled);
void OLED_Send_Pixel(char portnum,unsigned long color);
unsigned long OLED_Draw_Text_Small(char portnum,char *pStr, unsigned long posx, unsigned long posy, unsigned long color_f, unsigned long color_b);
unsigned long OLED_Draw_Text_Medium(char portnum,char *pStr, unsigned long posx, unsigned long posy, unsigned long color_f, unsigned long color_b);
unsigned long OLED_Draw_Text_Large(char *pStr, unsigned long posx, unsigned long posy, unsigned long color_f, unsigned long color_b);
void OLED_Draw_Char(char portnum,char ch, unsigned long posx, unsigned long posy, unsigned long color_f, unsigned long color_b, unsigned long scale);
void OLED_Draw_Dot(char portnum,signed long x, signed long y, signed long size, unsigned long color);
void OLED_Clear_Screen(char portnum,unsigned long color);
void OLED_Fill_Rect(char portnum,signed long x0, signed long y0, signed long xsize, signed long ysize, unsigned long color);
char OLED_Num4_to_Char(unsigned long num4);
unsigned long OLED_Draw_Hex(char portnum,unsigned long bitlen, unsigned long hex, unsigned long posx, unsigned long posy, unsigned long color_f, unsigned long color_b);
unsigned long OLED_Make_Color(unsigned long num);
//
void OLED_printf_Font(unsigned long font);
void OLED_printf_Color(unsigned long color_f, unsigned long color_b); // corrected 2011.03.20 MM
void OLED_printf_Position(unsigned long posx, unsigned long posy);
void OLED_printf(char portnum,const char *format, ...);

void OLED_move_position(char portnum,int x,int y);

#endif // __OLED_H__

//=========================================================
// End of Program
//=========================================================
