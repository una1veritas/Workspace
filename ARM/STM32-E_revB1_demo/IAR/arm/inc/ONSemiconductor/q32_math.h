/* ----------------------------------------------------------------------------
 * Copyright (c) 2009 - 2012 Semiconductor Components Industries, LLC
 * (d/b/a ON Semiconductor). All Rights Reserved.
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor. The
 * terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_math.h
 * - Header file for math library
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_MATH_H
#define Q32_MATH_H

#include <stdint.h>

/* ----------------------------------------------------------------------------
 * Firmware Math Library Version Code
 * ------------------------------------------------------------------------- */
#define MATH_FW_VER_MAJOR                0x02
#define MATH_FW_VER_MINOR                0x00
#define MATH_FW_VER_REVISION             0x00

#define MATH_FW_VER                      ((MATH_FW_VER_MAJOR << 12) | \
                                          (MATH_FW_VER_MINOR << 8)  | \
                                          (MATH_FW_VER_REVISION))

extern const int16_t q32_MathLib_Version;

/* ----------------------------------------------------------------------------
 * Math library defines
 * ------------------------------------------------------------------------- */
#define MAX_FRAC32 0x7FFFFFFF
#define MIN_FRAC32 0x80000000

/* ----------------------------------------------------------------------------
 * Math library function prototypes
 * ------------------------------------------------------------------------- */

/* Floating-Point32_t functions */

extern float Math_ExpAvg(float alpha, float x, float y1);

extern float Math_AttackRelease(float a, float b, float x, float y1);

extern float Math_LinearInterp(float x0, float x1, float y0, float y1, float x);

extern void Math_SingleVar_Reg(float* x, float* y, uint32_t N, float* a);

/* Fixed-Point32_t functions */

extern int32_t Math_Mult_frac32(int32_t x, int32_t y);

extern int32_t Math_Add_frac32(int32_t x, int32_t y);

extern int32_t Math_Sub_frac32(int32_t x, int32_t y);

extern int32_t Math_ExpAvg_frac32(int32_t alpha, int32_t x, int32_t y1);

extern int32_t Math_AttackRelease_frac32(int32_t a, int32_t b, int32_t x,
                                         int32_t y1);

extern int32_t Math_LinearInterp_frac32(int32_t y0, int32_t y1, int32_t x);

#endif  /* Q32_MATH_H */
