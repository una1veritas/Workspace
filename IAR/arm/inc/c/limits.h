/* limits.h standard header -- 8-bit version */
/* Copyright 2003-2010 IAR Systems AB. */
#ifndef _LIMITS
#define _LIMITS

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>

_C_STD_BEGIN

#if __AEABI_PORTABILITY_INTERNAL_LEVEL
  #ifndef __AEABI_PORTABLE
    #define __AEABI_PORTABLE
  #endif
        /* MULTIBYTE PROPERTIES */
  _C_LIB_DECL
  _DLIB_CONST_ATTR extern int const __aeabi_MB_LEN_MAX;
  _END_C_LIB_DECL
  #define MB_LEN_MAX (_CSTD __aeabi_MB_LEN_MAX)

#else /* __AEABI_PORTABILITY_INTERNAL_LEVEL */
        /* MULTIBYTE PROPERTIES */
  #define MB_LEN_MAX  _MBMAX

#endif /* __AEABI_PORTABILITY_INTERNAL_LEVEL */



        /* CHAR PROPERTIES */
#define CHAR_BIT    __CHAR_BITS__

#define CHAR_MAX    __CHAR_MAX__
#define CHAR_MIN    __CHAR_MIN__

        /* INT PROPERTIES */
#define INT_MAX     __SIGNED_INT_MAX__
#define INT_MIN     __SIGNED_INT_MIN__
#define UINT_MAX    __UNSIGNED_INT_MAX__

        /* SIGNED CHAR PROPERTIES */
#define SCHAR_MAX   __SIGNED_CHAR_MAX__
#define SCHAR_MIN   __SIGNED_CHAR_MIN__

        /* SHORT PROPERTIES */
#define SHRT_MAX    __SIGNED_SHORT_MAX__
#define SHRT_MIN    __SIGNED_SHORT_MIN__

        /* LONG PROPERTIES */
#define LONG_MAX    __SIGNED_LONG_MAX__
#define LONG_MIN    __SIGNED_LONG_MIN__

        /* UNSIGNED PROPERTIES */
#define UCHAR_MAX   __UNSIGNED_CHAR_MAX__
#define USHRT_MAX   __UNSIGNED_SHORT_MAX__
#define ULONG_MAX   __UNSIGNED_LONG_MAX__

        /* LONG LONG PROPERTIES */
#if _DLIB_ADD_C99_SYMBOLS
  #ifdef _LONGLONG
    #define LLONG_MIN __SIGNED_LONG_LONG_MIN__
    #define LLONG_MAX __SIGNED_LONG_LONG_MAX__
    #define ULLONG_MAX __UNSIGNED_LONG_LONG_MAX__
  #endif /* _LONGLONG */
#endif /* _DLIB_ADD_C99_SYMBOLS */

_C_STD_END

#endif /* _LIMITS */

#ifdef _STD_USING
  #if __AEABI_PORTABILITY_INTERNAL_LEVEL
    using _CSTD __aeabi_MB_LEN_MAX;
  #endif
#endif /* _STD_USING */
/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
