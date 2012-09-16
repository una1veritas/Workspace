/* float.h standard header -- IEEE 754 version */
/* Copyright 2003-2010 IAR Systems AB. */
#ifndef _FLOAT
#define _FLOAT

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>
_C_STD_BEGIN

                /* TYPE DEFINITIONS */

#if __SHORT_SIZE__ != 2
  #error "Float implementation assumes short is 2 bytes"
#endif

                /* COMMON PROPERTIES */

#define FLT_RADIX       2
#define FLT_ROUNDS      1 /* Towards nearest */

#if _DLIB_ADD_C99_SYMBOLS
  #define FLT_EVAL_METHOD 0 /* All operations are performed in own type */

                            /*
                             * If converting a floating-point number to
                             * and rhen back from a decimal number with
                             * DECIMAL_DIG digits then no information
                             * should be lost.
                             */

  #if __LONG_DOUBLE_SIZE__ == 8
    #define DECIMAL_DIG 17
  #else
    #define DECIMAL_DIG 10
  #endif
#endif /* _DLIB_ADD_C99_SYMBOLS */


/* Setup 4 byte floating point values */

#ifdef _FP4_BE_SETUP
  #define _FP4_MANT_DIG    _FP4_BE_MANT_DIG
  #define _FP4_DIG         _FP4_BE_DIG
  #define _FP4_MIN_EXP     _FP4_BE_MIN_EXP
  #define _FP4_MIN_10_EXP  _FP4_BE_MIN_10_EXP
  #define _FP4_MAX_EXP     _FP4_BE_MAX_EXP
  #define _FP4_MAX_10_EXP  _FP4_BE_MAX_10_EXP
  #define _FP4_MAX         _FP4_BE_MAX
  #define _FP4_EPSILON     _FP4_BE_EPSILON
  #define _FP4_MIN         _FP4_BE_MIN
#else
  #define _FP4_MANT_DIG    24
  #define _FP4_DIG         6
  #define _FP4_MIN_EXP     -125
  #define _FP4_MIN_10_EXP  -37
  #define _FP4_MAX_EXP     128
  #define _FP4_MAX_10_EXP  38
  #define _FP4_MAX         3.40282347e38   /* 0x1.FFFFFEp127  */
  #define _FP4_EPSILON     1.19209290e-7   /* 0x1.0p-23  */
  #define _FP4_MIN         1.17549436e-38  /* 0x1.0p-126 */
#endif


/* Setup 8 byte floating point values */

#ifdef _FP8_BE_SETUP
  #define _FP8_MANT_DIG    _FP8_BE_MANT_DIG
  #define _FP8_DIG         _FP8_BE_DIG
  #define _FP8_MIN_EXP     _FP8_BE_MIN_EXP
  #define _FP8_MIN_10_EXP  _FP8_BE_MIN_10_EXP
  #define _FP8_MAX_EXP     _FP8_BE_MAX_EXP
  #define _FP8_MAX_10_EXP  _FP8_BE_MAX_10_EXP
  #define _FP8_MAX         _FP8_BE_MAX
  #define _FP8_EPSILON     _FP8_BE_EPSILON
  #define _FP8_MIN         _FP8_BE_MIN
#else
  #define _FP8_MANT_DIG    53
  #define _FP8_DIG         15
  #define _FP8_MIN_EXP     -1021
  #define _FP8_MIN_10_EXP  -307
  #define _FP8_MAX_EXP     1024
  #define _FP8_MAX_10_EXP  308
  #define _FP8_MAX         1.7976931348623157e308  /* 0x1.FFFFFFFFFFFFFp1023  */
  #define _FP8_EPSILON     2.2204460492503131e-16  /* 0x1.0p-52  */
  #define _FP8_MIN         2.2250738585072014e-308 /* 0x1.0p-1022 */
#endif


                /* float properties */
#if __FLOAT_SIZE__ != 4
  #error "Float size must be 4 bytes"
#endif
#define FLT_MANT_DIG    _FP4_MANT_DIG
#define FLT_DIG         _FP4_DIG
#define FLT_MIN_EXP     _FP4_MIN_EXP
#define FLT_MIN_10_EXP  _FP4_MIN_10_EXP
#define FLT_MAX_EXP     _FP4_MAX_EXP
#define FLT_MAX_10_EXP  _FP4_MAX_10_EXP
#define FLT_MAX         _GLUE(_FP4_MAX, f)
#define FLT_EPSILON     _GLUE(_FP4_EPSILON, f)
#define FLT_MIN         _GLUE(_FP4_MIN, f)

                /* double properties */
#if __DOUBLE_SIZE__ == 4
  #define DBL_MANT_DIG    _FP4_MANT_DIG
  #define DBL_DIG         _FP4_DIG
  #define DBL_MIN_EXP     _FP4_MIN_EXP
  #define DBL_MIN_10_EXP  _FP4_MIN_10_EXP
  #define DBL_MAX_EXP     _FP4_MAX_EXP
  #define DBL_MAX_10_EXP  _FP4_MAX_10_EXP
  #define DBL_MAX         _FP4_MAX
  #define DBL_EPSILON     _FP4_EPSILON
  #define DBL_MIN         _FP4_MIN
#elif __DOUBLE_SIZE__ == 8
  #define DBL_MANT_DIG    _FP8_MANT_DIG
  #define DBL_DIG         _FP8_DIG
  #define DBL_MIN_EXP     _FP8_MIN_EXP
  #define DBL_MIN_10_EXP  _FP8_MIN_10_EXP
  #define DBL_MAX_EXP     _FP8_MAX_EXP
  #define DBL_MAX_10_EXP  _FP8_MAX_10_EXP
  #define DBL_MAX         _FP8_MAX
  #define DBL_EPSILON     _FP8_EPSILON
  #define DBL_MIN         _FP8_MIN
#else
  #error "Double type must be either 4 or 8 bytes"
#endif

                /* long double properties */
#if __LONG_DOUBLE_SIZE__ != __DOUBLE_SIZE__
  #error "Long double size must be the same as double size"
#endif
#define LDBL_MANT_DIG   DBL_MANT_DIG
#define LDBL_DIG        DBL_DIG
#define LDBL_MIN_EXP    DBL_MIN_EXP
#define LDBL_MIN_10_EXP DBL_MIN_10_EXP
#define LDBL_MAX_EXP    DBL_MAX_EXP
#define LDBL_MAX_10_EXP DBL_MAX_10_EXP
#define LDBL_MAX        _GLUE(DBL_MAX, l)
#define LDBL_EPSILON    _GLUE(DBL_EPSILON, l)
#define LDBL_MIN        _GLUE(DBL_MIN, l)

_C_STD_END
#endif /* _FLOAT */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
