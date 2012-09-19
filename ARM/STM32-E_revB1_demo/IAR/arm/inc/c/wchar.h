/* wchar.h standard header */
/* Copyright 2003-2010 IAR Systems AB.  */
#ifndef _WCHAR
#define _WCHAR

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>
#include <ysizet.h>

_C_STD_BEGIN

/* Consistency check */
#if !_DLIB_WIDE_CHARACTERS
  #error "This library configuration does not support wide characters."
#endif

/* Module consistency. */
#pragma rtmodel="__dlib_file_descriptor",_STRINGIFY(_DLIB_FILE_DESCRIPTOR)

/* Support for portable C++ object model. */
#if __AEABI_PORTABILITY_INTERNAL_LEVEL
  #ifndef __AEABI_PORTABLE
    #define __AEABI_PORTABLE
  #endif
#endif /* __AEABI_PORTABILITY_INTERNAL_LEVEL */


                /* MACROS */
#ifndef NULL
  #define NULL  _NULL
#endif /* NULL */

#define WCHAR_MIN       _WCMIN
#define WCHAR_MAX       _WCMAX
#define WEOF    ((_CSTD wint_t)(-1))

#if _WCMAX < __UNSIGNED_SHORT_MAX__
  #error "<wchart.h> wchar_t is too small."
#endif

                /* TYPE DEFINITIONS */
typedef _Mbstatet mbstate_t;

struct tm;

#ifndef _WCHART
  #define _WCHART
  typedef _Wchart wchar_t;
#endif /* _WCHART */

#ifndef _WINTT
  #define _WINTT
  typedef _Wintt wint_t;
#endif /* _WINT */

_C_LIB_DECL
                /* stdio DECLARATIONS */
#if _DLIB_FILE_DESCRIPTOR
  __ATTRIBUTES wint_t fgetwc(_Filet *);
  __ATTRIBUTES wchar_t * fgetws(wchar_t *_Restrict, int, _Filet *_Restrict);
  __ATTRIBUTES wint_t fputwc(wchar_t, _Filet *);
  __ATTRIBUTES int fputws(const wchar_t *_Restrict, _Filet *_Restrict);
  __ATTRIBUTES int fwide(_Filet *, int);
  __ATTRIBUTES int fwprintf(_Filet *_Restrict, const wchar_t *_Restrict, ...);
  __ATTRIBUTES int fwscanf(_Filet *_Restrict, const wchar_t *_Restrict, ...);
  __ATTRIBUTES wint_t getwc(_Filet *);
  __ATTRIBUTES wint_t putwc(wchar_t, _Filet *);
  __ATTRIBUTES wint_t ungetwc(wint_t, _Filet *);
  __ATTRIBUTES int vfwprintf(_Filet *_Restrict, const wchar_t *_Restrict,
                             __Va_list);
  #if _DLIB_ADD_C99_SYMBOLS
    __ATTRIBUTES int vfwscanf(_Filet *_Restrict, const wchar_t *_Restrict,
                              __Va_list);
  #endif /* _DLIB_ADD_C99_SYMBOLS */

#endif /* _DLIB_FILE_DESCRIPTOR */

__ATTRIBUTES wint_t getwchar(void);
__ATTRIBUTES wint_t __ungetwchar(wint_t);
__ATTRIBUTES wint_t putwchar(wchar_t);
__ATTRIBUTES int swprintf(wchar_t *_Restrict, size_t, 
                          const wchar_t *_Restrict, ...);
__ATTRIBUTES int swscanf(const wchar_t *_Restrict,
                         const wchar_t *_Restrict, ...);
__ATTRIBUTES int vswprintf(wchar_t *_Restrict, size_t,
                           const wchar_t *_Restrict, __Va_list);
__ATTRIBUTES int vwprintf(const wchar_t *_Restrict, __Va_list);
#if _DLIB_ADD_C99_SYMBOLS
  __ATTRIBUTES int vswscanf(const wchar_t *_Restrict, const wchar_t *_Restrict,
                            __Va_list);
  __ATTRIBUTES int vwscanf(const wchar_t *_Restrict, __Va_list);
#endif /* _DLIB_ADD_C99_SYMBOLS */
__ATTRIBUTES int wprintf(const wchar_t *_Restrict, ...);
__ATTRIBUTES int wscanf(const wchar_t *_Restrict, ...);

                /* stdlib DECLARATIONS */
__ATTRIBUTES size_t mbrlen(const char *_Restrict, size_t,
                           mbstate_t *_Restrict);
__ATTRIBUTES size_t mbrtowc(wchar_t *_Restrict, const char *, size_t,
                            mbstate_t *_Restrict);
__ATTRIBUTES size_t mbsrtowcs(wchar_t *_Restrict, const char **_Restrict,
                              size_t, mbstate_t *_Restrict);
__ATTRIBUTES int mbsinit(const mbstate_t *);
__ATTRIBUTES size_t wcrtomb(char *_Restrict, wchar_t, mbstate_t *_Restrict);
__ATTRIBUTES size_t wcsrtombs(char *_Restrict, const wchar_t **_Restrict,
                              size_t, mbstate_t *_Restrict);
__ATTRIBUTES long wcstol(const wchar_t *_Restrict, wchar_t **_Restrict, int);
__ATTRIBUTES unsigned long wcstoul(const wchar_t *_Restrict,
                                   wchar_t **_Restrict, int);

#if _DLIB_ADD_C99_SYMBOLS
  #ifdef _LONGLONG
    #pragma language=save
    #pragma language=extended
    __ATTRIBUTES _Longlong wcstoll(const wchar_t *_Restrict, 
                                   wchar_t **_Restrict, int);
    __ATTRIBUTES _ULonglong wcstoull(const wchar_t *_Restrict, 
                                     wchar_t **_Restrict, int);
    #pragma language=restore
  #endif
#endif /* _DLIB_ADD_C99_SYMBOLS */

                /* string DECLARATIONS */
__ATTRIBUTES wchar_t * wcscat(wchar_t *_Restrict, const wchar_t *_Restrict);
__ATTRIBUTES int wcscmp(const wchar_t *, const wchar_t *);
__ATTRIBUTES int wcscoll(const wchar_t *, const wchar_t *);
__ATTRIBUTES wchar_t * wcscpy(wchar_t *_Restrict, const wchar_t *_Restrict);
__ATTRIBUTES size_t wcscspn(const wchar_t *, const wchar_t *);
__ATTRIBUTES size_t wcslen(const wchar_t *);
__ATTRIBUTES wchar_t * wcsncat(wchar_t *_Restrict, const wchar_t *_Restrict, 
                               size_t);
__ATTRIBUTES int wcsncmp(const wchar_t *, const wchar_t *, size_t);
__ATTRIBUTES wchar_t * wcsncpy(wchar_t *_Restrict, const wchar_t *_Restrict,
                               size_t);
__ATTRIBUTES size_t wcsspn(const wchar_t *, const wchar_t *);
__ATTRIBUTES wchar_t * wcstok(wchar_t *_Restrict, const wchar_t *_Restrict,
                              wchar_t **_Restrict);
__ATTRIBUTES size_t wcsxfrm(wchar_t *_Restrict, const wchar_t *_Restrict, 
                            size_t);
__ATTRIBUTES int wmemcmp(const wchar_t *, const wchar_t *, size_t);
__ATTRIBUTES wchar_t * wmemcpy(wchar_t *_Restrict, const wchar_t *_Restrict, 
                               size_t);
__ATTRIBUTES wchar_t * wmemmove(wchar_t *, const wchar_t *, size_t);
__ATTRIBUTES wchar_t * wmemset(wchar_t *, wchar_t, size_t);

                /* time DECLARATIONS */
__ATTRIBUTES size_t wcsftime(wchar_t *_Restrict, size_t,
                             const wchar_t *_Restrict, 
                             const struct tm *_Restrict);


__ATTRIBUTES wint_t btowc(int);
#if _DLIB_ADD_C99_SYMBOLS
  __ATTRIBUTES float wcstof(const wchar_t *_Restrict, wchar_t **_Restrict);
  __ATTRIBUTES long double wcstold(const wchar_t *_Restrict,
                                   wchar_t **_Restrict);
#endif /* _DLIB_ADD_C99_SYMBOLS */
__ATTRIBUTES double wcstod(const wchar_t *_Restrict, wchar_t **_Restrict);
__ATTRIBUTES int wctob(wint_t);

__ATTRIBUTES wint_t __iar_Btowc(int);
__ATTRIBUTES int __iar_Wctob(wint_t);
__ATTRIBUTES double __iar_WStod(const wchar_t *, wchar_t **, long);
__ATTRIBUTES float __iar_WStof(const wchar_t *, wchar_t **, long);
__ATTRIBUTES long double __iar_WStold(const wchar_t *, wchar_t **, long);
__ATTRIBUTES unsigned long __iar_WStoul(const wchar_t *, wchar_t **, int);
__ATTRIBUTES _Longlong __iar_WStoll(const wchar_t *, wchar_t **, int);
__ATTRIBUTES _ULonglong __iar_WStoull(const wchar_t *, wchar_t **, int);

__ATTRIBUTES wchar_t * __iar_Wmemchr(const wchar_t *, wchar_t, size_t);
__ATTRIBUTES wchar_t * __iar_Wcschr(const wchar_t *, wchar_t);
__ATTRIBUTES wchar_t * __iar_Wcspbrk(const wchar_t *, const wchar_t *);
__ATTRIBUTES wchar_t * __iar_Wcsrchr(const wchar_t *, wchar_t);
__ATTRIBUTES wchar_t * __iar_Wcsstr(const wchar_t *, const wchar_t *);
_END_C_LIB_DECL

/* IAR, can't use the Dinkum stratagem for wmemchr,... */

#ifdef __cplusplus
  _EXTERN_CPP
  __ATTRIBUTES const wchar_t * wmemchr(const wchar_t *, wchar_t, size_t);
  __ATTRIBUTES const wchar_t * wcschr(const wchar_t *, wchar_t);
  __ATTRIBUTES const wchar_t * wcspbrk(const wchar_t *, const wchar_t *);
  __ATTRIBUTES const wchar_t * wcsrchr(const wchar_t *, wchar_t);
  __ATTRIBUTES const wchar_t * wcsstr(const wchar_t *, const wchar_t *);
  __ATTRIBUTES wchar_t * wmemchr(wchar_t *, wchar_t, size_t);
  __ATTRIBUTES wchar_t * wcschr(wchar_t *, wchar_t);
  __ATTRIBUTES wchar_t * wcspbrk(wchar_t *, const wchar_t *);
  __ATTRIBUTES wchar_t * wcsrchr(wchar_t *, wchar_t);
  __ATTRIBUTES wchar_t * wcsstr(wchar_t *, const wchar_t *);
  _END_EXTERN_CPP
#else /* !__cplusplus */
  __ATTRIBUTES wchar_t * wmemchr(const wchar_t *, wchar_t, size_t);
  __ATTRIBUTES wchar_t * wcschr(const wchar_t *, wchar_t);
  __ATTRIBUTES wchar_t * wcspbrk(const wchar_t *, const wchar_t *);
  __ATTRIBUTES wchar_t * wcsrchr(const wchar_t *, wchar_t);
  __ATTRIBUTES wchar_t * wcsstr(const wchar_t *, const wchar_t *);
#endif /* __cplusplus */

#if !defined(_NO_DEFINITIONS_IN_HEADER_FILES) && !__AEABI_PORTABILITY_INTERNAL_LEVEL
  #ifdef __cplusplus
  _EXTERN_CPP
                /* INLINES AND OVERLOADS, FOR C++ */

    inline const wchar_t * wmemchr(const wchar_t *_S, wchar_t _C, size_t _N)
    {
      return (__iar_Wmemchr(_S, _C, _N));
    }

    inline const wchar_t * wcschr(const wchar_t *_S, wchar_t _C)
    {
      return (__iar_Wcschr(_S, _C));
    }

    inline const wchar_t * wcspbrk(const wchar_t *_S, const wchar_t *_P)
    {
      return (__iar_Wcspbrk(_S, _P));
    }

    inline const wchar_t * wcsrchr(const wchar_t *_S, wchar_t _C)
    {
      return (__iar_Wcsrchr(_S, _C));
    }

    inline const wchar_t * wcsstr(const wchar_t *_S, const wchar_t *_P)
    {
      return (__iar_Wcsstr(_S, _P));
    }

    inline wchar_t * wmemchr(wchar_t *_S, wchar_t _C, size_t _N)
    {
      return (__iar_Wmemchr(_S, _C, _N));
    }

    inline wchar_t * wcschr(wchar_t *_S, wchar_t _C)
    {
      return (__iar_Wcschr(_S, _C));
    }

    inline wchar_t * wcspbrk(wchar_t *_S, const wchar_t *_P)
    {
      return (__iar_Wcspbrk(_S, _P));
    }

    inline wchar_t * wcsrchr(wchar_t *_S, wchar_t _C)
    {
      return (__iar_Wcsrchr(_S, _C));
    }

    inline wchar_t * wcsstr(wchar_t *_S, const wchar_t *_P)
    {
      return (__iar_Wcsstr(_S, _P));
    }
  _END_EXTERN_CPP
  #else /* __cplusplus */
    #pragma inline
    wchar_t * wmemchr(const wchar_t *_S, wchar_t _C, size_t _N)
    {
      return (__iar_Wmemchr(_S, _C, _N));
    }

    #pragma inline
    wchar_t * wcschr(const wchar_t *_S, wchar_t _C)
    {
      return (__iar_Wcschr(_S, _C));
    }

    #pragma inline
    wchar_t * wcspbrk(const wchar_t *_S, const wchar_t *_P)
    {
      return (__iar_Wcspbrk(_S, _P));
    }

    #pragma inline
    wchar_t * wcsrchr(const wchar_t *_S, wchar_t _C)
    {
      return (__iar_Wcsrchr(_S, _C));
    }

    #pragma inline
    wchar_t * wcsstr(const wchar_t *_S, const wchar_t *_P)
    {
      return (__iar_Wcsstr(_S, _P));
    }
  #endif /* __cplusplus */

  #pragma inline
  wint_t btowc(int _C)
  {       /* convert single byte to wide character */
    return (__iar_Btowc(_C));
  }

  #if _DLIB_ADD_C99_SYMBOLS
    #pragma inline
    float wcstof(const wchar_t *_S,
                 wchar_t **_Endptr)
    {       /* convert wide string to double */
      return (__iar_WStof(_S, _Endptr, 0));
    }

    #pragma inline
    long double wcstold(const wchar_t *_S,
                        wchar_t **_Endptr)
    {       /* convert wide string to double */
      return (__iar_WStold(_S, _Endptr, 0));
    }

    #ifdef _LONGLONG
      #pragma language=save
      #pragma language=extended
      #pragma inline
       _Longlong wcstoll(const wchar_t *_Restrict _S, 
                         wchar_t **_Restrict _Endptr, int _Base)
       {
	return (__iar_WStoll(_S, _Endptr, _Base));
       }

      #pragma inline
      _ULonglong wcstoull(const wchar_t *_Restrict _S, 
                          wchar_t **_Restrict _Endptr, int _Base)
      {
	return (__iar_WStoull(_S, _Endptr, _Base));
      }
      #pragma language=restore
    #endif /*_LONGLONG */

  #endif /* _DLIB_ADD_C99_SYMBOLS */

  #pragma inline
  double wcstod(const wchar_t *_S,
                wchar_t **_Endptr)
  {       /* convert wide string to double */
    return (__iar_WStod(_S, _Endptr, 0));
  }


  #pragma inline
  unsigned long wcstoul(const wchar_t *_S,
                        wchar_t **_Endptr, int _Base)
  {       /* convert wide string to unsigned long */
    return (__iar_WStoul(_S, _Endptr, _Base));
  }

  #pragma inline
  int wctob(wint_t _Wc)
  {       /* convert wide character to single byte */
    return (__iar_Wctob(_Wc));
  }

#endif /* _NO_DEFINITIONS_IN_HEADER_FILES */

_C_STD_END
#endif /* _WCHAR */

#if defined(_STD_USING) && defined(__cplusplus)
  using _CSTD mbstate_t; using _CSTD tm; using _CSTD wint_t;

  using _CSTD btowc; using _CSTD getwchar;
  using _CSTD mbrlen; using _CSTD mbrtowc; using _CSTD mbsrtowcs;
  using _CSTD mbsinit; using _CSTD putwchar;
  using _CSTD swprintf; using _CSTD swscanf; 
  using _CSTD vswprintf; using _CSTD vwprintf;
  using _CSTD wcrtomb; using _CSTD wprintf; using _CSTD wscanf;
  using _CSTD wcsrtombs; using _CSTD wcstol; using _CSTD wcscat;
  using _CSTD wcschr; using _CSTD wcscmp; using _CSTD wcscoll;
  using _CSTD wcscpy; using _CSTD wcscspn; using _CSTD wcslen;
  using _CSTD wcsncat; using _CSTD wcsncmp; using _CSTD wcsncpy;
  using _CSTD wcspbrk; using _CSTD wcsrchr; using _CSTD wcsspn;
  using _CSTD wcstod; using _CSTD wcstoul; using _CSTD wcsstr;
  using _CSTD wcstok; using _CSTD wcsxfrm; using _CSTD wctob;
  using _CSTD wmemchr; using _CSTD wmemcmp; using _CSTD wmemcpy;
  using _CSTD wmemmove; using _CSTD wmemset; using _CSTD wcsftime;
  #if _DLIB_ADD_C99_SYMBOLS
    using _CSTD vswscanf; using _CSTD vwscanf; using _CSTD wcstof; 
    using _CSTD wcstold;
    #ifdef _LONGLONG
      using _CSTD wcstoll; using _CSTD wcstoull;
    #endif
  #endif /* _DLIB_ADD_C99_SYMBOLS */
  #if _DLIB_FILE_DESCRIPTOR
    using _CSTD fgetwc; using _CSTD fgetws; using _CSTD fputwc;
    using _CSTD fputws; using _CSTD fwide; using _CSTD fwprintf;
    using _CSTD fwscanf; using _CSTD getwc; using _CSTD putwc; 
    using _CSTD ungetwc; using _CSTD vfwprintf; 
    #if _DLIB_ADD_C99_SYMBOLS
      using _CSTD vfwscanf; 
    #endif /* _DLIB_ADD_C99_SYMBOLS */
  #endif /* _DLIB_FILE_DESCRIPTOR */

  using _CSTD __Va_list;
  using _CSTD __iar_Btowc; using _CSTD __iar_Wctob; using _CSTD __iar_WStod;
  using _CSTD __iar_WStof; using _CSTD __iar_WStold; using _CSTD __iar_WStoul;
  using _CSTD __iar_WStoll; using _CSTD __iar_WStoull; 

#endif /* defined(_STD_USING) && defined(__cplusplus) */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
