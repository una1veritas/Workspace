/* time.h standard header */
/* Copyright 2003-2010 IAR Systems AB. */
#ifndef _TIME
#define _TIME

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>
#include <ysizet.h>

_C_STD_BEGIN

#if __AEABI_PORTABILITY_INTERNAL_LEVEL
  #ifndef __AEABI_PORTABLE
    #define __AEABI_PORTABLE
  #endif

  _C_LIB_DECL
  _DLIB_CONST_ATTR extern int const __aeabi_CLOCKS_PER_SEC;
  _END_C_LIB_DECL
  #define CLOCKS_PER_SEC  (_CSTD __aeabi_CLOCKS_PER_SEC)

  #if _DLIB_TIME_USES_64
    #error("_DLIB_TIME_USES_64 cannot be used together with _AEABI_PORTABILITY_LEVEL" )
  #endif
#else /* __AEABI_PORTABILITY_INTERNAL_LEVEL */
  #define CLOCKS_PER_SEC  1
#endif /* __AEABI_PORTABILITY_INTERNAL_LEVEL */


#if  _DLIB_SUPPORT_FOR_AEABI
  #if _DLIB_TIME_USES_LONG
    typedef signed long __time32_t;
    typedef signed long clock_t;
  #else /* !_DLIB_TIME_USES_UNSIGNED_LONG */
    typedef unsigned int __time32_t;
    typedef unsigned int clock_t;
  #endif /* _DLIB_TIME_USES_UNSIGNED_LONG */
#else /* _DLIB_SUPPORT_FOR_AEABI */
  typedef signed long __time32_t;
  typedef signed long clock_t;
#endif /* _DLIB_SUPPORT_FOR_AEABI */

#if _DLIB_TIME_ALLOW_64
  #pragma language=save
  #pragma language=extended
  typedef signed long long __time64_t;
  #pragma language=restore
#endif /* _DLIB_TIME_ALLOW_64 */

#if defined(_DLIB_TIME_USES_64)
  #if !_DLIB_TIME_ALLOW_64
    #error("_DLIB_TIME_USES_64 requires _DLIB_TIME_ALLOW_64")
  #endif
  #if defined(_NO_DEFINITIONS_IN_HEADER_FILES)
    #error("_DLIB_TIME_USES_64 cannot be used with _NO_DEFINITIONS_IN_HEADER_FILES")
  #endif
#else   
  #define _DLIB_TIME_USES_64 _DLIB_TIME_USES_64_DEFAULT
#endif  /* _DLIB_TIME_USES_64 */

#if _DLIB_TIME_USES_64
  typedef __time64_t time_t;
#else
  typedef __time32_t time_t;
#endif

struct tm
{       /* date and time components */
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;
#if  _DLIB_SUPPORT_FOR_AEABI
  int __BSD_bug_filler1;
  int __BSD_bug_filler2;
#endif /* _DLIB_SUPPORT_FOR_AEABI */
};

#ifndef NULL
  #define NULL   _NULL
#endif /* NULL */


_EXTERN_C       /* low-level functions */
__ATTRIBUTES time_t time(time_t *);
__ATTRIBUTES __time32_t __time32(__time32_t *);
#if _DLIB_TIME_ALLOW_64
  __ATTRIBUTES __time64_t __time64(__time64_t *);
#endif /* _DLIB_TIME_ALLOW_64 */
_END_EXTERN_C

_C_LIB_DECL     /* declarations */
__ATTRIBUTES char * asctime(const struct tm *);
__ATTRIBUTES clock_t clock(void);
__ATTRIBUTES char * ctime(const time_t *);
__EFF_NE __ATTRIBUTES double difftime(time_t, time_t);
__ATTRIBUTES struct tm * gmtime(const time_t *);
__ATTRIBUTES struct tm * localtime(const time_t *);
__ATTRIBUTES time_t mktime(struct tm *);

__ATTRIBUTES char * __ctime32(const __time32_t *);
__EFF_NE __ATTRIBUTES double __difftime32(__time32_t, __time32_t);
__ATTRIBUTES struct tm * __gmtime32(const __time32_t *);
__ATTRIBUTES struct tm * __localtime32(const __time32_t *);
__ATTRIBUTES __time32_t __mktime32(struct tm *);
#if _DLIB_TIME_ALLOW_64
  __ATTRIBUTES char * __ctime64(const __time64_t *);
  __EFF_NE __ATTRIBUTES double __difftime64(__time64_t, __time64_t);
  __ATTRIBUTES struct tm * __gmtime64(const __time64_t *);
  __ATTRIBUTES struct tm * __localtime64(const __time64_t *);
  __ATTRIBUTES __time64_t __mktime64(struct tm *);
#endif /* _DLIB_TIME_ALLOW_64 */
__ATTRIBUTES size_t strftime(char *_Restrict, size_t, const char *_Restrict,
                             const struct tm *_Restrict);
_END_C_LIB_DECL

#if !defined(_NO_DEFINITIONS_IN_HEADER_FILES) && !__AEABI_PORTABILITY_INTERNAL_LEVEL
/* C inline definitions */

  #pragma inline=forced
  time_t time(time_t *p)
  {
    #if _DLIB_TIME_USES_64
      return __time64(p);
    #else
      return __time32(p);
    #endif
  }

  #pragma inline=forced
  char * ctime(const time_t *p)
  {
    #if _DLIB_TIME_USES_64
      return __ctime64(p);
    #else
      return __ctime32(p);
    #endif
  }

  #pragma inline=forced
  double difftime(time_t t1, time_t t2)
  {
    #if _DLIB_TIME_USES_64
      return __difftime64(t1, t2);
    #else
      return __difftime32(t1, t2);
    #endif
  }

  #pragma inline=forced
  struct tm * gmtime(const time_t *p)
  {
    #if _DLIB_TIME_USES_64
      return __gmtime64(p);
    #else
      return __gmtime32(p);
    #endif
  }

  #pragma inline=forced
  struct tm * localtime(const time_t *p)
  {
    #if _DLIB_TIME_USES_64
      return __localtime64(p);
    #else
      return __localtime32(p);
    #endif
  }

  #pragma inline=forced
  time_t mktime(struct tm *p)
  {
    #if _DLIB_TIME_USES_64
      return __mktime64(p);
    #else
      return __mktime32(p);
    #endif
  }

#endif /* !defined(_NO_DEFINITIONS_IN_HEADER_FILES) && !__AEABI_PORTABILITY_INTERNAL_LEVEL */


_C_STD_END
#endif /* _TIME */

#if defined(_STD_USING) && defined(__cplusplus)
  using _CSTD clock_t;
  using _CSTD time_t;
  using _CSTD tm;
  using _CSTD asctime; using _CSTD clock; using _CSTD ctime;
  using _CSTD difftime; using _CSTD gmtime; using _CSTD localtime;
  using _CSTD mktime; using _CSTD strftime; using _CSTD time;

  using _CSTD __time32_t;
  using _CSTD __ctime32; using _CSTD __difftime32; 
  using _CSTD __gmtime32; using _CSTD __localtime32;
  using _CSTD __mktime32; using _CSTD __time32;
  #if _DLIB_TIME_ALLOW_64
    using _CSTD __time64_t;
    using _CSTD __ctime64; using _CSTD __difftime64; 
    using _CSTD __gmtime64; using _CSTD __localtime64;
    using _CSTD __mktime64; using _CSTD __time64;
  #endif /* _DLIB_TIME_ALLOW_64 */

  #if __AEABI_PORTABILITY_INTERNAL_LEVEL
    using _CSTD __aeabi_CLOCKS_PER_SEC;
  #endif
#endif /* defined(_STD_USING) && defined(__cplusplus) */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
