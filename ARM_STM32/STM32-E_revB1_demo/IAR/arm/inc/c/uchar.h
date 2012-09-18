/* uchar.h added header for TR 19769 */
/* Copyright 2009-2010 IAR Systems AB.  */
#ifndef _UCHAR
#define _UCHAR

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>
#include <ysizet.h>

_C_STD_BEGIN

                /* macros */
#ifndef NULL
  #define NULL  _NULL
#endif /* NULL */

/* Values of char16_t are UTF-16 encoded */
#define __STDC_UTF_16__
/* Values of char32_t are UTF-32 encoded */
#define __STDC_UTF_32__

_C_LIB_DECL
                /* TYPE DEFINITIONS */
typedef _Mbstatet mbstate_t;

typedef unsigned short char16_t;
typedef unsigned long char32_t;

                /* declarations */
__ATTRIBUTES size_t mbrtoc16(char16_t *_Restrict, const char *_Restrict,
                             size_t, mbstate_t *_Restrict);
__ATTRIBUTES size_t c16rtomb(char *_Restrict, char16_t,
                             mbstate_t *_Restrict);

__ATTRIBUTES size_t mbrtoc32(char32_t *_Restrict, const char *_Restrict,
                             size_t, mbstate_t *_Restrict);
__ATTRIBUTES size_t c32rtomb(char *_Restrict, char32_t,
                             mbstate_t *_Restrict);
_END_C_LIB_DECL

_C_STD_END
#endif /* _UCHAR */

#if defined(_STD_USING) && defined(__cplusplus)
  using _CSTD char16_t; using _CSTD char32_t; using _CSTD mbstate_t;
  using _CSTD mbrtoc16; using _CSTD c16rtomb;
  using _CSTD mbrtoc32; using _CSTD c32rtomb;
#endif /* defined(_STD_USING) */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
