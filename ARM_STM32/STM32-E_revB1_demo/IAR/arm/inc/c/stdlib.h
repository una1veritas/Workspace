/* stdlib.h standard header */
/* Copyright 2005-2010 IAR Systems AB. */

#ifndef _STDLIB
#define _STDLIB

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>
#include <ysizet.h>
#include <xencoding_limits.h>

/* Module consistency. */
#pragma rtmodel="__dlib_full_locale_support", \
  _STRINGIFY(_DLIB_FULL_LOCALE_SUPPORT)

_C_STD_BEGIN

_C_LIB_DECL
extern int __aeabi_MB_CUR_MAX(void);
_END_C_LIB_DECL

#if __AEABI_PORTABILITY_INTERNAL_LEVEL
  #ifndef __AEABI_PORTABLE
    #define __AEABI_PORTABLE
  #endif

  #define MB_CUR_MAX    (_CSTD __aeabi_MB_CUR_MAX())

#else /* __AEABI_PORTABILITY_INTERNAL_LEVEL  */

  #define MB_CUR_MAX    (_ENCODING_CUR_MAX)

#endif /*  __AEABI_PORTABILITY_INTERNAL_LEVEL */

                /* MACROS */
#ifndef NULL
  #define NULL  _NULL
#endif /* NULL */

#define EXIT_FAILURE    1
#define EXIT_SUCCESS    0

#if _ILONG
  #define RAND_MAX      0x3fffffff
#else /* _ILONG */
  #define RAND_MAX      0x7fff
#endif /* _ILONG */

                /* TYPE DEFINITIONS */
#ifndef _WCHART
  #define _WCHART
  typedef _Wchart wchar_t;
#endif /* _WCHART */

typedef struct
{       /* result of int divide */
  int quot;
  int rem;
} div_t;

typedef struct
{       /* result of long divide */
  long quot;
  long rem;
} ldiv_t;

#if _DLIB_ADD_C99_SYMBOLS
  #ifdef _LONGLONG
    #pragma language=save
    #pragma language=extended
    typedef struct
    {     /* result of long long divide */
      _Longlong quot;
      _Longlong rem;
    } lldiv_t;
    #pragma language=restore
  #endif
#endif /* _DLIB_ADD_C99_SYMBOLS */

                /* DECLARATIONS */
_EXTERN_C /* low-level functions */
__ATTRIBUTES int atexit(void (*)(void));
#if _DLIB_ADD_C99_SYMBOLS
  __ATTRIBUTES_NORETURN void _Exit(int) _NO_RETURN;
#endif /* _DLIB_ADD_C99_SYMBOLS */
__ATTRIBUTES_NORETURN void exit(int) _NO_RETURN;
__ATTRIBUTES char * getenv(const char *);
__ATTRIBUTES int system(const char *);
_END_EXTERN_C

_C_LIB_DECL
             __ATTRIBUTES_NORETURN void abort(void) _NO_RETURN;
__EFF_NE     __ATTRIBUTES int abs(int);
             __ATTRIBUTES void * calloc(size_t, size_t);
__EFF_NE     __ATTRIBUTES div_t div(int, int);
             __ATTRIBUTES void free(void *);
__EFF_NE     __ATTRIBUTES long labs(long);
__EFF_NE     __ATTRIBUTES ldiv_t ldiv(long, long);
#if _DLIB_ADD_C99_SYMBOLS
  #ifdef _LONGLONG
    #pragma language=save
    #pragma language=extended
    __EFF_NE __ATTRIBUTES long long llabs(long long);
    __EFF_NE __ATTRIBUTES lldiv_t lldiv(long long, long long);
    #pragma language=restore
  #endif
#endif /* _DLIB_ADD_C99_SYMBOLS */
             __ATTRIBUTES void * malloc(size_t);
__EFF_NW1    __ATTRIBUTES int mblen(const char *, size_t);
__EFF_NR1NW2 __ATTRIBUTES size_t mbstowcs(wchar_t *_Restrict, 
                                          const char *_Restrict, size_t);
__EFF_NR1NW2 __ATTRIBUTES int mbtowc(wchar_t *_Restrict, const char *_Restrict, 
                                     size_t);
             __ATTRIBUTES int rand(void);
             __ATTRIBUTES void srand(unsigned int);
             __ATTRIBUTES void * realloc(void *, size_t);
__EFF_NW1NR2 __ATTRIBUTES long strtol(const char *_Restrict, 
                                      char **_Restrict, int);
__EFF_NW1NR2 __ATTRIBUTES unsigned long strtoul(const char *, char **, int);
__EFF_NR1NW2 __ATTRIBUTES size_t wcstombs(char *_Restrict, 
                                          const wchar_t *_Restrict, size_t);
__EFF_NR1    __ATTRIBUTES int wctomb(char *, wchar_t);
#if _DLIB_ADD_C99_SYMBOLS
  #ifdef _LONGLONG
    #pragma language=save
    #pragma language=extended
    __EFF_NW1NR2 __ATTRIBUTES long long strtoll(const char *, char **, int);
    __EFF_NW1NR2 __ATTRIBUTES unsigned long long strtoull(const char *, 
                                                          char **, int);
    #pragma language=restore
  #endif
#endif /* _DLIB_ADD_C99_SYMBOLS */

#if __AEABI_PORTABILITY_INTERNAL_LEVEL == 0

#if __MULTIPLE_HEAPS__

#pragma language=save
#pragma language=extended

#define __HEAP_MEM_HELPER1__(M, I)                              \
__ATTRIBUTES void M##_free(void M *);                            \
__ATTRIBUTES void M * M##_malloc(M##_size_t);                    \
__ATTRIBUTES void M * M##_calloc(M##_size_t, M##_size_t);        \
__ATTRIBUTES void M * M##_realloc(void M *, M##_size_t);
__HEAP_MEMORY_LIST1__()
#undef __HEAP_MEM_HELPER1__

#ifndef _NO_DEFINITIONS_IN_HEADER_FILES
#ifndef _DO_NOT_INLINE_MALLOC

#ifndef __DEF_HEAP_MEM__
  #define __DEF_HEAP_MEM__ __DEF_PTR_MEM__
#endif

#pragma inline
void free(void * _P)
{
  _GLUE(__DEF_HEAP_MEM__,_free((void __DEF_HEAP_MEM__ *)_P));
}
#pragma inline
void * malloc(size_t _S)
{
  return _GLUE(__DEF_HEAP_MEM__,_malloc(_S));

}
#pragma inline
void * realloc(void * _P, size_t _S)
{
  return _GLUE(__DEF_HEAP_MEM__,_realloc((void __DEF_HEAP_MEM__ *)_P, _S));
}
#pragma inline
void * calloc(size_t _N, size_t _S)
{
  return _GLUE(__DEF_HEAP_MEM__,_calloc(_N, _S));
}

#endif /* _DO_NOT_INLINE_MALLOC */
#endif /* _NO_DEFINITIONS_IN_HEADER_FILES */

#pragma language=restore

#endif /* __MULTIPLE_HEAPS__ */

#endif /* __AEABI_PORTABILITY_INTERNAL_LEVEL == 0 */


__EFF_NW1NR2 __ATTRIBUTES unsigned long __iar_Stoul(const char *, char **, 
                                                    int);
__EFF_NW1NR2 __ATTRIBUTES float         __iar_Stof(const char *, char **, long);
__EFF_NW1NR2 __ATTRIBUTES double        __iar_Stod(const char *, char **, long);
__EFF_NW1NR2 __ATTRIBUTES long double   __iar_Stold(const char *, char **, 
                                                      long);
__EFF_NW1NR2 __ATTRIBUTES long          __iar_Stolx(const char *, char **, int, 
                                                    int *);
__EFF_NW1NR2 __ATTRIBUTES unsigned long __iar_Stoulx(const char *, char **,
                                                     int, int *);
__EFF_NW1NR2 __ATTRIBUTES float         __iar_Stofx(const char *, char **, 
                                                    long, int *);
__EFF_NW1NR2 __ATTRIBUTES double        __iar_Stodx(const char *, char **, 
                                                    long, int *);
__EFF_NW1NR2 __ATTRIBUTES long double   __iar_Stoldx(const char *, char **, 
                                                     long, int *);
#ifdef _LONGLONG
  #pragma language=save
  #pragma language=extended
  __EFF_NW1NR2 __ATTRIBUTES _Longlong   __iar_Stoll(const char *, char **, 
                                                    int);
  __EFF_NW1NR2 __ATTRIBUTES _ULonglong   __iar_Stoull(const char *, char **, 
                                                      int);
  __EFF_NW1NR2 __ATTRIBUTES _Longlong    __iar_Stollx(const char *, char **, 
                                                      int, int *);
  __EFF_NW1NR2 __ATTRIBUTES _ULonglong   __iar_Stoullx(const char *, char **, 
                                                       int, int *);
  #pragma language=restore
#endif

_EXTERN_C
typedef int _Cmpfun(const void *, const void *);
_END_EXTERN_C
__EFF_NW1NW2 __ATTRIBUTES_CAN_THROW void * bsearch(const void *, 
                                                   const void *, size_t,
                                                   size_t, _Cmpfun *);
             __ATTRIBUTES_CAN_THROW void qsort(void *, size_t, size_t, 
                                               _Cmpfun *);
             __ATTRIBUTES_CAN_THROW void __qsortbbl(void *, size_t, size_t, 
                                                    _Cmpfun *);
__EFF_NW1    __ATTRIBUTES double atof(const char *);
__EFF_NW1    __ATTRIBUTES int atoi(const char *);
__EFF_NW1    __ATTRIBUTES long atol(const char *);
#if _DLIB_ADD_C99_SYMBOLS
  #ifdef _LONGLONG
    #pragma language=save
    #pragma language=extended
    __EFF_NW1 __ATTRIBUTES long long atoll(const char *);
    #pragma language=restore
  #endif
  __EFF_NW1NR2 __ATTRIBUTES float strtof(const char *_Restrict, 
                                         char **_Restrict);
  __EFF_NW1NR2 __ATTRIBUTES long double strtold(const char *, char **);
#endif /* _DLIB_ADD_C99_SYMBOLS */
__EFF_NW1NR2 __ATTRIBUTES double strtod(const char *_Restrict, 
                                        char **_Restrict);
             __ATTRIBUTES size_t __iar_Mbcurmax(void);

__EFF_NE     __ATTRIBUTES int __iar_DLib_library_version(void);
_END_C_LIB_DECL

#ifdef __cplusplus
  _EXTERN_CPP
  #if _HAS_STRICT_LINKAGE
    typedef int _Cmpfun2(const void *, const void *);
    int atexit(void (*)(void));
    __EFF_NW1NW2 void * bsearch(const void *, const void *, size_t, size_t, 
                                _Cmpfun2 *);
    void qsort(void *, size_t, size_t, _Cmpfun2 *);
  #endif /* _HAS_STRICT_LINKAGE */

  __EFF_NE long abs(long);
  __EFF_NE ldiv_t div(long, long);
  _END_EXTERN_CPP
#endif /* __cplusplus */


#ifndef _NO_DEFINITIONS_IN_HEADER_FILES
  _EXTERN_C
  typedef void _Atexfun(void);
  _END_EXTERN_C
  #if _HAS_STRICT_LINKAGE && defined(__cplusplus)
  _EXTERN_CPP
    #pragma inline
    int atexit(void (*_Pfn)(void))
    {     // register a function to call at exit
      return (atexit((_Atexfun *)_Pfn));
    }

    #pragma inline
    void * bsearch(const void *_Key, const void *_Base,
                   size_t _Nelem, size_t _Size, _Cmpfun2 *_Cmp)
    {     // search by binary chop
      return (bsearch(_Key, _Base, _Nelem, _Size, (_Cmpfun *)_Cmp));
    }

    #pragma inline
    void qsort(void *_Base, size_t _Nelem, size_t _Size, _Cmpfun2 *_Cmp)
    {     // sort
      qsort(_Base, _Nelem, _Size, (_Cmpfun *)_Cmp);
    }
  _END_EXTERN_CPP
  #endif /* _HAS_STRICT_LINKAGE */

  #if !__AEABI_PORTABILITY_INTERNAL_LEVEL
                /* INLINES, FOR C and C++ */
    #pragma inline=no_body
    double atof(const char *_S)
    {      /* convert string to double */
      return (__iar_Stod(_S, 0, 0));
    }

    #pragma inline=no_body
    int atoi(const char *_S)
    {      /* convert string to int */
      return ((int)__iar_Stoul(_S, 0, 10));
    }

    #pragma inline=no_body
    long atol(const char *_S)
    {      /* convert string to long */
      return ((long)__iar_Stoul(_S, 0, 10));
    }

    #if _DLIB_ADD_C99_SYMBOLS
      #ifdef _LONGLONG
        #pragma language=save
        #pragma language=extended
        #pragma inline=no_body
        long long atoll(const char *_S)
        {      /* convert string to long long */
          #if __LONG_LONG_SIZE__ == __LONG_SIZE__
            return ((long long)__iar_Stoul(_S, 0, 10));
          #else
            return ((long long)__iar_Stoull(_S, 0, 10));
          #endif
        }
        #pragma language=restore
      #endif
    #endif /* _DLIB_ADD_C99_SYMBOLS */

    #pragma inline=no_body
    double strtod(const char *_Restrict _S, char **_Restrict _Endptr)
    {      /* convert string to double, with checking */
      return (__iar_Stod(_S, _Endptr, 0));
    }

    #if _DLIB_ADD_C99_SYMBOLS
      #pragma inline=no_body
      float strtof(const char *_Restrict _S, char **_Restrict _Endptr)
      {      /* convert string to float, with checking */
        return (__iar_Stof(_S, _Endptr, 0));
      }

      #pragma inline=no_body
      long double strtold(const char *_Restrict _S, char **_Restrict _Endptr)
      {      /* convert string to long double, with checking */
        return (__iar_Stold(_S, _Endptr, 0));
      }
    #endif /* _DLIB_ADD_C99_SYMBOLS */

    #pragma inline=no_body
    long strtol(const char *_Restrict _S, char **_Restrict _Endptr, 
                int _Base)
    {      /* convert string to unsigned long, with checking */
      return (__iar_Stolx(_S, _Endptr, _Base, 0));
    }

    #pragma inline=no_body
    unsigned long strtoul(const char *_Restrict _S, char **_Restrict _Endptr, 
                          int _Base)
    {      /* convert string to unsigned long, with checking */
      return (__iar_Stoul(_S, _Endptr, _Base));
    }

    #if _DLIB_ADD_C99_SYMBOLS
      #ifdef _LONGLONG
        #pragma language=save
        #pragma language=extended
        #pragma inline=no_body
        long long strtoll(const char *_Restrict _S, char **_Restrict _Endptr,
                          int _Base)
        {      /* convert string to long long, with checking */
          #if __LONG_LONG_SIZE__ == __LONG_SIZE__
          return ((long long)__iar_Stolx(_S, _Endptr, _Base, 0));
          #else
            return (__iar_Stoll(_S, _Endptr, _Base));
          #endif
        }

        #pragma inline=no_body
        unsigned long long strtoull(const char *_Restrict _S, 
                                    char **_Restrict _Endptr, int _Base)
        {      /* convert string to unsigned long long, with checking */
          #if __LONG_LONG_SIZE__ == __LONG_SIZE__
            return ((unsigned long long)__iar_Stoul(_S, _Endptr, _Base));
          #else
            return (__iar_Stoull(_S, _Endptr, _Base));
          #endif
        }
        #pragma language=restore
      #endif /* _LONGLONG */
    #endif /* _DLIB_ADD_C99_SYMBOLS */

  #endif /* __AEABI_PORTABILITY_INTERNAL_LEVEL */

  #pragma inline=no_body
  int abs(int i)
  {      /* compute absolute value of int argument */
    return (i < 0 ? -i : i);
  }

  #pragma inline=no_body
  long labs(long i)
  {      /* compute absolute value of long argument */
    return (i < 0 ? -i : i);
  }

  #if _DLIB_ADD_C99_SYMBOLS
    #ifdef _LONGLONG
      #pragma language=save
      #pragma language=extended
      #pragma inline=no_body
      long long llabs(long long i)
      {      /* compute absolute value of long long argument */
        return (i < 0 ? -i : i);
      }
      #pragma language=restore
    #endif
  #endif /* _DLIB_ADD_C99_SYMBOLS */

  #ifdef __cplusplus
  _EXTERN_CPP
    #pragma inline=forced
    inline long abs(long _X)     /* OVERLOADS */
    {     /* compute abs */
      return (labs(_X));
    }

    #pragma inline=forced
    inline ldiv_t div(long _X, long _Y)
    {     /* compute quotient and remainder */
      return (ldiv(_X, _Y));
    }

    #ifdef _LONGLONG
      #pragma language=save
      #pragma language=extended
      #pragma inline=forced
      inline long long abs(long long _X)     /* OVERLOADS */
      {     /* compute abs */
        return (llabs(_X));
      }

      #pragma inline=forced
      inline lldiv_t div(long long _X, long long _Y)
      {     /* compute quotient and remainder */
        return (lldiv(_X, _Y));
      }
      #pragma language=restore
    #endif
  _END_EXTERN_CPP
  #endif /* __cplusplus */
#endif /* _NO_DEFINITIONS_IN_HEADER_FILES */

_C_STD_END
#endif /* _STDLIB */

#if defined(_STD_USING) && defined(__cplusplus)
  using _CSTD div_t; using _CSTD ldiv_t;

  using _CSTD abort; using _CSTD abs; using _CSTD atexit;
  using _CSTD atof; using _CSTD atoi; using _CSTD atol;
  using _CSTD bsearch; using _CSTD calloc; using _CSTD div;
  using _CSTD exit; using _CSTD free; using _CSTD getenv;
  using _CSTD labs; using _CSTD ldiv; using _CSTD malloc;
  using _CSTD mblen; using _CSTD mbstowcs; using _CSTD mbtowc;
  using _CSTD qsort; using _CSTD rand; using _CSTD realloc;
  using _CSTD srand; using _CSTD strtod;using _CSTD strtol; 
  using _CSTD strtoul; using _CSTD system;
  using _CSTD wcstombs; using _CSTD wctomb;
  using _CSTD __qsortbbl;
  #if _DLIB_ADD_C99_SYMBOLS
    using _CSTD strtof; using _CSTD strtold;
    using _CSTD _Exit;
    #ifdef _LONGLONG
      using _CSTD lldiv_t;

      using _CSTD atoll; using _CSTD llabs; using _CSTD lldiv;
      using _CSTD strtoll; using _CSTD strtoull;
    #endif
  #endif /* _DLIB_ADD_C99_SYMBOLS */
#endif /* defined(_STD_USING) && defined(__cplusplus) */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
