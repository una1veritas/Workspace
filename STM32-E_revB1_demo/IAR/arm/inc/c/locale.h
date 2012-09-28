/* locale.h standard header */
/* Copyright 2003-2010 IAR Systems AB. */
#ifndef _LOCALE
#define _LOCALE

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>

_C_STD_BEGIN


/* Module consistency. */
#pragma rtmodel="__dlib_full_locale_support", \
  _STRINGIFY(_DLIB_FULL_LOCALE_SUPPORT)

                /* MACROS */
#ifndef NULL
  #define NULL   _NULL
#endif /* NULL */


                /* LOCALE CATEGORY INDEXES */
#define _LC_COLLATE             0
#define _LC_CTYPE               1
#define _LC_MONETARY            2
#define _LC_NUMERIC             3
#define _LC_TIME                4
#define _LC_MESSAGE             5
                /* ADD YOURS HERE, THEN UPDATE _NCAT */
#define _NCAT                   6       /* one more than last index */

                /* LOCALE CATEGORY MASKS */
#define _CATMASK(n)     (1 << (n))

#define _M_COLLATE      _CATMASK(_LC_COLLATE)
#define _M_CTYPE        _CATMASK(_LC_CTYPE)
#define _M_MONETARY     _CATMASK(_LC_MONETARY)
#define _M_NUMERIC      _CATMASK(_LC_NUMERIC)
#define _M_TIME         _CATMASK(_LC_TIME)
#define _M_MESSAGE      _CATMASK(_LC_MESSAGE)
#define _M_MESSAGES     _M_MESSAGE
#define _M_ALL          (_CATMASK(_NCAT) - 1)



#if __AEABI_PORTABILITY_INTERNAL_LEVEL
  #ifndef __AEABI_PORTABLE
    #define __AEABI_PORTABLE
  #endif

  _C_LIB_DECL
  _DLIB_CONST_ATTR extern int const __aeabi_LC_COLLATE;
  _DLIB_CONST_ATTR extern int const __aeabi_LC_CTYPE;
  _DLIB_CONST_ATTR extern int const __aeabi_LC_MONETARY;
  _DLIB_CONST_ATTR extern int const __aeabi_LC_NUMERIC;
  _DLIB_CONST_ATTR extern int const __aeabi_LC_TIME;
  _DLIB_CONST_ATTR extern int const __aeabi_LC_MESSAGE;
  _DLIB_CONST_ATTR extern int const __aeabi_LC_MESSAGES;
  _DLIB_CONST_ATTR extern int const __aeabi_LC_ALL;
  _END_C_LIB_DECL
  #define LC_COLLATE    _CSTD __aeabi_LC_COLLATE
  #define LC_CTYPE      _CSTD __aeabi_LC_CTYPE
  #define LC_MONETARY   _CSTD __aeabi_LC_MONETARY
  #define LC_NUMERIC    _CSTD __aeabi_LC_NUMERIC
  #define LC_TIME       _CSTD __aeabi_LC_TIME
  #define LC_MESSAGE    _CSTD __aeabi_LC_MESSAGE
  #define LC_MESSAGES   LC_MESSAGE
  #define LC_ALL        _CSTD __aeabi_LC_ALL

#else /* __AEABI_PORTABILITY_INTERNAL_LEVEL */

                /* LOCALE CATEGORY HANDLES */
  #define LC_COLLATE      _CATMASK(_LC_COLLATE)
  #define LC_CTYPE        _CATMASK(_LC_CTYPE)
  #define LC_MONETARY     _CATMASK(_LC_MONETARY)
  #define LC_NUMERIC      _CATMASK(_LC_NUMERIC)
  #define LC_TIME         _CATMASK(_LC_TIME)
  #define LC_MESSAGE      _CATMASK(_LC_MESSAGE)
  #define LC_MESSAGES     LC_MESSAGE
  #define LC_ALL          (_CATMASK(_NCAT) - 1)

#endif /* __AEABI_PORTABILITY_INTERNAL_LEVEL */

#define _X_COLLATE      LC_COLLATE
#define _X_CTYPE        LC_CTYPE
#define _X_MONETARY     LC_MONETARY
#define _X_NUMERIC      LC_NUMERIC
#define _X_TIME         LC_TIME
#define _X_MESSAGES     LC_MESSAGES
#define _X_MAX          _X_MESSAGES     /* highest real category */


                /* MACROS FOR LOCKING GLOBAL LOCALES */
#if _GLOBAL_LOCALE
  #define _Locklocale()         __iar_Locksyslock_Locale()
  #define _Unlocklocale()       __iar_Unlocksyslock_Locale()

#else /* _GLOBAL_LOCALE */
  #define _Locklocale()         (void)0
  #define _Unlocklocale()       (void)0
#endif /* _GLOBAL_LOCALE */

                /* TYPE DEFINITIONS */

#if __AEABI_PORTABILITY_INTERNAL_LEVEL

  #include <xlocale_alconv.h>

#else /* __AEABI_PORTABILITY_INTERNAL_LEVEL */

  #include <xlocale_lconv.h>

#endif /* __AEABI_PORTABILITY_INTERNAL_LEVEL */

                /* DECLARATIONS */
_C_LIB_DECL
#if _DLIB_SUPPORT_FOR_AEABI
__ATTRIBUTES struct __aeabi_lconv *__aeabi_localeconv(void);
#endif
#if !__AEABI_PORTABILITY_INTERNAL_LEVEL
__ATTRIBUTES struct lconv *localeconv(void);
#endif
#if _DLIB_FULL_LOCALE_SUPPORT
  __ATTRIBUTES char *setlocale(int, const char *);
#endif
_END_C_LIB_DECL

_C_STD_END
#endif /* _LOCALE */

#ifdef _STD_USING
  #if _DLIB_SUPPORT_FOR_AEABI
    using _CSTD __aeabi_lconv; using _CSTD __aeabi_localeconv;
  #endif
  #if __AEABI_PORTABILITY_INTERNAL_LEVEL
    using _CSTD __aeabi_LC_COLLATE;  using _CSTD __aeabi_LC_CTYPE;
    using _CSTD __aeabi_LC_MONETARY; using _CSTD __aeabi_LC_NUMERIC;
    using _CSTD __aeabi_LC_TIME;     using _CSTD __aeabi_LC_MESSAGE;
    using _CSTD __aeabi_LC_MESSAGES; using _CSTD __aeabi_LC_ALL;
  #endif
  #if !__AEABI_PORTABILITY_INTERNAL_LEVEL
    using _CSTD lconv; using _CSTD localeconv;
  #endif
  #if _DLIB_FULL_LOCALE_SUPPORT
    using _CSTD setlocale;
  #endif
#endif /* _STD_USING */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
