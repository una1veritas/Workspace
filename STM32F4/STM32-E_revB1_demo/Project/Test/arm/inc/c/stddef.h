/* stddef.h standard header */
/* Copyright 2009-2010 IAR Systems AB. */
#ifndef _STDDEF
#define _STDDEF

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>
#include <ysizet.h>

_C_STD_BEGIN

                /* macros */
#ifndef NULL
  #define NULL   _NULL
#endif /* NULL */

#ifndef offsetof
  #define offsetof(T, member)     (__INTADDR__((&((T *)0)->member)))
#endif /* offsetof */

                /* type definitions */
#if !defined(_PTRDIFF_T) && !defined(_PTRDIFFT)
  #define _PTRDIFF_T
  #define _PTRDIFFT
  #define _STD_USING_PTRDIFF_T
  typedef _Ptrdifft ptrdiff_t;
#endif /* !defined(_PTRDIFF_T) && !defined(_PTRDIFFT) */

#ifndef _WCHART
  #define _WCHART
  typedef _Wchart wchar_t;
#endif /* _WCHART */
_C_STD_END
#endif /* _STDDEF */

#if defined(_STD_USING) && defined(__cplusplus)
  #ifdef _STD_USING_PTRDIFF_T
    using _CSTD ptrdiff_t;
  #endif /* _STD_USING_PTRDIFF_T */

  #ifdef _STD_USING_SIZE_T
    using _CSTD size_t;
  #endif /* _STD_USING_SIZE_T */
#endif /* defined(_STD_USING) && defined(__cplusplus) */


/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
