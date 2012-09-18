/* yvals.h internal configuration header file. */
/* Copyright 2001-2010 IAR Systems AB. */

#ifndef _YVALS
#define _YVALS

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>

                /* Convenience macros */
#define _GLUE_B(x,y) x##y
#define _GLUE(x,y) _GLUE_B(x,y)

#define _GLUE3_B(x,y,z) x##y##z
#define _GLUE3(x,y,z) _GLUE3_B(x,y,z)

#define _STRINGIFY_B(x) #x
#define _STRINGIFY(x) _STRINGIFY_B(x)

/* Used to refer to '__aeabi' symbols in the library. */ 
#define _ABINAME(x) _GLUE_B(__aeabi_, x)

                /* Versions */
#define _CPPLIB_VER     504

#ifndef __IAR_SYSTEMS_LIB__
  #define __IAR_SYSTEMS_LIB__ 5
#endif

#if (__IAR_SYSTEMS_ICC__ < 8) || (__IAR_SYSTEMS_ICC__ > 8)
  #error "<yvals.h>  compiled with wrong (version of IAR) compiler"
#endif

/*
 * Support for some C99 or other symbols
 *
 * This setting makes available some macros, functions, etc that are
 * beneficial.
 *
 * Default is to include them.
 *
 * Disabling this in C++ mode will not compile (some C++ functions uses C99
 * functionality).
 */

#ifndef _DLIB_ADD_C99_SYMBOLS
  /* Default turned on when compiling C++, EC++, or C99. */
  #if defined(__cplusplus)
    #define _DLIB_ADD_C99_SYMBOLS 1
  #elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define _DLIB_ADD_C99_SYMBOLS 1
  #else
    #define _DLIB_ADD_C99_SYMBOLS 0
  #endif
#endif /* _DLIB_ADD_C99_SYMBOLS */

#ifndef _DLIB_ADD_EXTRA_SYMBOLS
  #define _DLIB_ADD_EXTRA_SYMBOLS 1
#endif /* _DLIB_ADD_EXTRA_SYMBOLS */

#ifdef __LONG_LONG_SIZE__
  #define _LONGLONG     long long
  #define _ULONGLONG    unsigned long long
  #define _LLONG_MAX    __SIGNED_LONG_LONG_MAX__
  #define _ULLONG_MAX   __UNSIGNED_LONG_LONG_MAX__
#endif /* __LONG_LONG_SIZE__ */

                /* Configuration */
#include <DLib_Defaults.h>

#define _HAS_PRAGMA_PRINTF_ARGS

#ifndef _NO_RETURN
  #define _NO_RETURN
#endif /* _NO_RETURN */

#if __AEABI_PORTABILITY_INTERNAL_LEVEL && !_DLIB_SUPPORT_FOR_AEABI
  #error "__AEABI_PORTABILITY_LEVEL != 0 needs a library built with _DLIB_SUPPORT_FOR_AEABI turned on"
#endif

                /* Floating-point */

/*
 * Whenever a floating-point type is equal to another, we try to fold those
 * two types into one. This means that if float == double then we fold float to
 * use double internally. Example sinf(float) will use _Sin(double, uint).
 *
 * _X_FNAME is a redirector for internal support routines. The X can be
 *          D (double), F (float), or L (long double). It redirects by using
 *          another prefix. Example calls to Dtest will be __iar_Dtest,
 *          __iar_FDtest, or __iarLDtest.
 * _X_FUN   is a redirector for functions visible to the customer. As above, the
 *          X can be D, F, or L. It redirects by using another suffix. Example
 *          calls to sin will be sin, sinf, or sinl.
 * _X_TYPE  The type that one type is folded to.
 * _X_PTRCAST is a redirector for a cast operation involving a pointer.
 * _X_CAST  is a redirector for a cast involving the float type.
 *
 * _FLOAT_IS_DOUBLE signals that all internal float routines aren't needed.
 * _LONG_DOUBLE_IS_DOUBLE signals that all internal long double routines
 *                        aren't needed.
 */
#ifndef _NO_FLOAT_FOLDING
  #if __FLOAT_SIZE__ == __DOUBLE_SIZE__
    #define _FLOAT_IS_DOUBLE
    #define _F_FNAME(fun) __iar_##fun
    #define _F_FUN(fun)   fun
    #define _F_TYPE       double
    #define _F_PTRCAST    (double *)
    #define _F_CAST       (double)
  #else
    #define _F_FNAME(fun) __iar_F##fun
    #define _F_FUN(fun)   fun##f
    #define _F_TYPE       float
    #define _F_PTRCAST
    #define _F_CAST
  #endif
  #if __LONG_DOUBLE_SIZE__ == __DOUBLE_SIZE__
    #define _LONG_DOUBLE_IS_DOUBLE
    #define _L_FNAME(fun) __iar_##fun
    #define _L_FUN(fun)   fun
    #define _L_TYPE       double
    #define _L_PTRCAST    (double *)
    #define _L_CAST       (double)
  #else
    #define _L_FNAME(fun) __iar_L##fun
    #define _L_FUN(fun)   fun##l
    #define _L_TYPE       long double
    #define _L_PTRCAST
    #define _L_CAST
  #endif
#else /* _NO_FLOAT_FOLDING */
  #define _F_FNAME(fun) __iar_F##fun
  #define _F_FUN(fun)   fun##f
  #define _F_TYPE       float
  #define _F_PTRCAST
  #define _F_CAST
  #define _L_FNAME(fun) __iar_L##fun
  #define _L_FUN(fun)   fun##l
  #define _L_TYPE       long double
  #define _L_PTRCAST
  #define _L_CAST
#endif /* !_NO_FLOAT_FOLDING */

#define _D_FNAME(fun) __iar_##fun
#define _D_FUN(fun)   fun
#define _D_TYPE       double

                /* NAMING PROPERTIES */
#define _HAS_STRICT_LINKAGE           0       /* extern "C" in function type */

/* Has support for fixed point types */
#ifndef _HAS_FIXED_POINT 
  #define _HAS_FIXED_POINT 0
#endif

/* Has support for secure functions (printf_s, scanf_s, etc) */
/* Will not compile if enabled */
#ifndef _HAS_SECURE
  #define _HAS_SECURE 0
#endif
#if _HAS_SECURE
  #define _SECURE_PARAM(x) , (x)
#else
  #define _SECURE_PARAM(x)
#endif  

/* Has support for complex C types */
#ifndef _HAS_DINKUM_COMPLEX
  #define _HAS_DINKUM_COMPLEX 1
#endif

/* If is Embedded C++ language */
#if defined(__cplusplus) && defined(__embedded_cplusplus)
  #define _IS_EMBEDDED 1
#else
  #define _IS_EMBEDDED 0
#endif

/* If is true C++ language */
#if defined(__cplusplus) && !defined(__embedded_cplusplus)
  #define _IS_CPP_LANGUAGE 1
#else
  #define _IS_CPP_LANGUAGE 0
#endif

/* True C++ language setup */
#if _IS_CPP_LANGUAGE
  /* Enables/Disables iterator debugging 
     (the setting must be the same for the whole application). */ 
  #ifndef _HAS_ITERATOR_DEBUGGING
    #define _HAS_ITERATOR_DEBUGGING 0
  #elif _HAS_ITERATOR_DEBUGGING != 0 && _HAS_ITERATOR_DEBUGGING != 1
    #error "Faulty value used"
  #endif
  /* Iterator debugging consistency. */
  #pragma rtmodel="__dlib_iterator_debugging", \
                  _STRINGIFY(_HAS_ITERATOR_DEBUGGING)

  #ifdef __EXCEPTIONS
    #define _HAS_EXCEPTIONS 1
  #else 
    #define _HAS_EXCEPTIONS 0
  #endif
  #define _HAS_NAMESPACE  1
  #define _HAS_IMMUTABLE_SETS 0
  #define _HAS_HASH_STATISTICS 0

  #if __MULTIPLE_HEAPS__
    #error "C++ doesn't support multiple heaps"
  #endif

  /* exports C names from std to global, else reversed */
  #define _STD_USING   

  /* These settings cannot be changed. */
  #define _HAS_GENERIC_TEMPLATES 1
  #define _HAS_TRADITIONAL_STL 0
  #define _HAS_TRADITIONAL_IOSTREAMS 0
  #define _HAS_STLPORT_EMULATION 0
  #define _HAS_TRADITIONAL_POS_TYPE 0
  #define _HAS_STRICT_CONFORMANCE 1
  #define _HAS_TRADITIONAL_ITERATORS 0
  /* #define _HAS_CONVENTIONAL_CLIB 0 */
  /* #define _HAS_POINTER_CLIB 0 */
  /* #define _HAS_DINKUM_CLIB 1 */
  /* #define _STD_LINKAGE defines C names as extern "C++" */
#else /*  !_IS_CPP_LANGUAGE */
  #define _HAS_EXCEPTIONS 0
  #define _HAS_NAMESPACE  0
  #define _HAS_GENERIC_TEMPLATES 0
#endif /*  _IS_CPP_LANGUAGE */

#ifdef __WCHAR_T
  #define _HAS_WCHAR_TYPE 1
#endif /* __WCHAR_T */

#if defined(__cplusplus)
  #ifndef __ARRAY_OPERATORS
    #error "<yvals.h> __ARRAY_OPERATORS not defined (c++)"
  #endif /* __ARRAY_OPERATORS */
#endif /* __cplusplus */

                /* NAMESPACE CONTROL */
#if defined(__cplusplus)
  #if _HAS_NAMESPACE
    #define _STD_BEGIN  namespace std {
    #define _STD_END    }
    #define _STD        ::std::

    #ifdef _STD_USING
      #define _C_STD_BEGIN    namespace std { /* only if *.c compiled as C++ */
      #define _C_STD_END      }
      #define _CSTD     ::std::
    #else /* _STD_USING */
      #define _GLOBAL_USING    /* *.h in global namespace, c* imports to std */

      #define _C_STD_BEGIN
      #define _C_STD_END
      #define _CSTD     ::
    #endif /* _STD_USING */

    #define _X_STD_BEGIN    namespace std {
    #define _X_STD_END      }
    #define _XSTD           ::std::

  #else /* _HAS_NAMESPACE */
    #define _STD_BEGIN
    #define _STD_END
    #define _STD        ::

    #define _C_STD_BEGIN
    #define _C_STD_END
    #define _CSTD       ::

    #define _X_STD_BEGIN
    #define _X_STD_END
    #define _XSTD       ::
  #endif /* _HAS_NAMESPACE */

  /* C has extern "C" linkage */
  #define _C_LIB_DECL         extern "C" {
  #define _END_C_LIB_DECL     }
  #define _EXTERN_C           extern "C" {
  #define _END_EXTERN_C       }
  #define _EXTERN_CPP         extern "C++" {
  #define _END_EXTERN_CPP     }
#else /* __cplusplus */
  #define _STD_BEGIN
  #define _STD_END
  #define _STD

  #define _C_STD_BEGIN
  #define _C_STD_END
  #define _CSTD

  #define _X_STD_BEGIN
  #define _X_STD_END
  #define _XSTD

  #define _C_LIB_DECL
  #define _END_C_LIB_DECL
  #define _EXTERN_C
  #define _END_EXTERN_C
  #define _EXTERN_CPP
  #define _END_EXTERN_CPP
#endif /* __cplusplus */

#ifdef __cplusplus
  _STD_BEGIN
  typedef bool _Bool;
  _STD_END
#endif /* __cplusplus */


#include <xencoding_limits.h>

_C_STD_BEGIN

                /* FLOATING-POINT PROPERTIES */

                /* float properties */
#if __FLOAT_SIZE__ == 4
  #define _FBIAS 0x7e    /* IEEE 754 float properties */
  #define _FOFF  7
  #define _FMANTISSA 23
  #if __LITTLE_ENDIAN__
    #define _F0    1
  #else
    #define _F0    0
  #endif
#else
  #error "<yvals.h> __FLOAT_SIZE__ not 4"
#endif /* __FLOAT_SIZE__ */

                /* double properties */
#if __DOUBLE_SIZE__ == 8
  #define _DBIAS 0x3fe   /* IEEE 754 double properties */
  #define _DOFF  4
  #define _LOFF  4
  #define _DMANTISSA 52
  #if __LITTLE_ENDIAN__
    #define _D0    3
  #else
    #define _D0    0
  #endif
#elif __DOUBLE_SIZE__ == 4
  #define _DBIAS _FBIAS
  #define _DOFF  7
  #define _LOFF  7
  #define _DMANTISSA _FMANTISSA
  #if __LITTLE_ENDIAN__
    #define _D0    1
  #else
    #define _D0    0
  #endif
#else
  #error "<yvals.h> __DOUBLE_SIZE__ not 4 or 8"
#endif /* __DOUBLE_SIZE__ */

                /* long double properties */
                /* (must be same as double) */
#define _DLONG 0
#define _LBIAS _DBIAS
#define _LMANTISSA _DMANTISSA

#if __LONG_DOUBLE_SIZE__ == 8
  #if __LITTLE_ENDIAN__
    #define _L0    3
  #else
    #define _L0    0
  #endif
#elif __LONG_DOUBLE_SIZE__ == 4
  #if __LITTLE_ENDIAN__
    #define _L0    1
  #else
    #define _L0    0
  #endif
#else
  #error "<yvals.h> __LONG_DOUBLE_SIZE__ not 4 or 8"
#endif /* __LONG_DOUBLE_SIZE__ */


                /* INTEGER PROPERTIES */
#define _C2             1       /* 0 if not 2's complement */
                                /* MB_LEN_MAX */
#define _MBMAX          _ENCODING_LEN_MAX

#define _MAX_EXP_DIG    8       /* for parsing numerics */
#define _MAX_INT_DIG    32
#define _MAX_SIG_DIG    36

#ifdef _LONGLONG
  #pragma language=save
  #pragma language=extended
  typedef _LONGLONG _Longlong;
  typedef _ULONGLONG _ULonglong;
  #pragma language=restore
#else /* _LONGLONG */
  typedef long _Longlong;
  typedef unsigned long _ULonglong;
  #define _LLONG_MAX  __SIGNED_LONG_MAX__
  #define _ULLONG_MAX __UNSIGNED_LONG_MAX__
#endif /* _LONGLONG */

#ifdef __cplusplus
  #define _WCHART
  typedef wchar_t _Wchart;
  typedef wchar_t _Wintt;
#else
  typedef __WCHAR_T_TYPE__ _Wchart;
  typedef __WCHAR_T_TYPE__ _Wintt;
#endif

#ifdef __SIGNED_WCHAR_T__
  #define _WCMIN  __WCHAR_T_MIN__
  #define _WIMIN  __WCHAR_T_MIN__
#else
  #define _WCMIN  0
  #define _WIMIN  0
#endif
#define _WCMAX  __WCHAR_T_MAX__
#define _WIMAX  __WCHAR_T_MAX__

#if __INT_SIZE__ == 2
  #define _ILONG 0
#elif __INT_SIZE__ == 4
  #define _ILONG 1
#else
  #error "__INT_SIZE__ must be 2 or 4"
#endif /* __INT_SIZE__ */

                /* POINTER PROPERTIES */
#define _NULL           0       /* 0L if pointer same as long */

typedef __PTRDIFF_T_TYPE__  _Ptrdifft;
typedef __SIZE_T_TYPE__     _Sizet;

/* IAR doesn't support restrict  */
#define _Restrict

                /* stdarg PROPERTIES */
#ifndef _VA_DEFINED
  #ifndef _VA_LIST_STACK_MEMORY_ATTRIBUTE
    #define _VA_LIST_STACK_MEMORY_ATTRIBUTE
  #endif

  typedef struct
  {
    char _VA_LIST_STACK_MEMORY_ATTRIBUTE *_Ap;
  } __Va_list;
#else /* _VA_DEFINED */
  typedef _VA_LIST __Va_list;
#endif /* !_VA_DEFINED */

_EXTERN_C
__ATTRIBUTES void __iar_Atexit(void (*)(void));
_END_EXTERN_C

#if _DLIB_SUPPORT_FOR_AEABI && !_DLIB_MBSTATET_USES_UNSIGNED_LONG
  typedef struct
  {       /* state of a multibyte translation */
    unsigned int _Wchar;
    unsigned int _State;
  } _Mbstatet;
  #if __INT_SIZE__ != 4
    #pragma error "sizeof int must be 4"
  #endif
#else /* _DLIB_SUPPORT_FOR_AEABI */
  typedef struct
  {       /* state of a multibyte translation */
    unsigned long _Wchar;      /* Used as an intermediary value (up to 32-bits) */
    unsigned long _State;      /* Used as a state value (only some bits used) */
  } _Mbstatet;
#endif /* _DLIB_SUPPORT_FOR_AEABI */


#if _DLIB_FILE_DESCRIPTOR
  typedef struct __FILE _Filet;
#endif

typedef struct
{       /* file position */
#if _DLIB_SUPPORT_FOR_AEABI
  _Longlong _Off;    /* can be system dependent */
#else
  long _Off;    /* can be system dependent */
#endif
  _Mbstatet _Wstate;
} _Fpost;

#ifndef _FPOSOFF
  #define _FPOSOFF(fp)  ((fp)._Off)
#endif

_C_STD_END

                /* THREAD AND LOCALE CONTROL */

#include <DLib_Threads.h>

#ifndef _MULTI_THREAD
  #define _MULTI_THREAD 0     /* 0 for no locks, 1 for multithreaded library */
#endif /* _MULTI_THREAD */
#ifndef _GLOBAL_LOCALE
  #define _GLOBAL_LOCALE  0       /* 0 for per-thread locales, 1 for shared */
#endif /* _GLOBAL_LOCALE */
#ifndef _FILE_OP_LOCKS
  #define _FILE_OP_LOCKS  0       /* 0 for no file atomic locks, 1 for atomic */
#endif /* _FILE_OP_LOCKS */

                /* THREAD-LOCAL STORAGE */
#ifndef _COMPILER_TLS
  #define _COMPILER_TLS   0       /* 1 if compiler supports TLS directly */
#endif /* _COMPILER_TLS */
#ifndef _TLS_QUAL
  #define _TLS_QUAL     /* TLS qualifier, such as __declspec(thread), if any */
#endif /* _TLS_QUAL */


                /* MULTITHREAD PROPERTIES */
#if _MULTI_THREAD
  _C_STD_BEGIN
  _EXTERN_C
  /* The lock interface for DLib to use. */ 
  __WEAK __ATTRIBUTES void __iar_Locksyslock_Locale(void);
  __WEAK __ATTRIBUTES void __iar_Locksyslock_Malloc(void);
  __WEAK __ATTRIBUTES void __iar_Locksyslock_Stream(void);
  __WEAK __ATTRIBUTES void __iar_Locksyslock_Debug(void);
  __WEAK __ATTRIBUTES void __iar_Locksyslock_StaticGuard(void);
  __WEAK __ATTRIBUTES void __iar_Locksyslock(unsigned int);
  __WEAK __ATTRIBUTES void __iar_Unlocksyslock_Locale(void);
  __WEAK __ATTRIBUTES void __iar_Unlocksyslock_Malloc(void);
  __WEAK __ATTRIBUTES void __iar_Unlocksyslock_Stream(void);
  __WEAK __ATTRIBUTES void __iar_Unlocksyslock_Debug(void);
  __WEAK __ATTRIBUTES void __iar_Unlocksyslock_StaticGuard(void);
  __WEAK __ATTRIBUTES void __iar_Unlocksyslock(unsigned int);

  __WEAK __ATTRIBUTES void __iar_Initdynamicfilelock(__iar_Rmtx *);
  __WEAK __ATTRIBUTES void __iar_Dstdynamicfilelock(__iar_Rmtx *);
  __WEAK __ATTRIBUTES void __iar_Lockdynamicfilelock(__iar_Rmtx *);
  __WEAK __ATTRIBUTES void __iar_Unlockdynamicfilelock(__iar_Rmtx *);
  _END_EXTERN_C
  _C_STD_END
#else /* _MULTI_THREAD */
  #define __iar_Locksyslock_Locale()        (void)0
  #define __iar_Locksyslock_Malloc()        (void)0
  #define __iar_Locksyslock_Stream()        (void)0
  #define __iar_Locksyslock_Debug()         (void)0
  #define __iar_Locksyslock_StaticGuard()   (void)0
  #define __iar_Locksyslock(x)              (void)0
  #define __iar_Unlocksyslock_Locale()      (void)0
  #define __iar_Unlocksyslock_Malloc()      (void)0
  #define __iar_Unlocksyslock_Stream()      (void)0
  #define __iar_Unlocksyslock_Debug()       (void)0
  #define __iar_Unlocksyslock_StaticGuard() (void)0
  #define __iar_Unlocksyslock(x)            (void)0
#endif /* _MULTI_THREAD */

                /* LOCK MACROS */
#define _LOCK_LOCALE       0
#define _LOCK_MALLOC       1
#define _LOCK_STREAM       2
#define _LOCK_DEBUG        3
#define _LOCK_STATIC_GUARD 4    /* Static once initialization */
#define _MAX_LOCK          5    /* one more than highest lock number */

#ifdef __cplusplus
  _STD_BEGIN
                // CLASS _Lockit
  class __iar_Lockit
  {     // Generic version
        // lock while object in existence -- MUST NEST
  public:
  #if _MULTI_THREAD
    explicit __iar_Lockit(int _Type)
      : _Locktype(_Type)
    {   // set the lock
      __iar_Locksyslock(_Locktype);
    }

    ~__iar_Lockit()
    {   // clear the lock
      __iar_Unlocksyslock(_Locktype);
    }

  private:
    int _Locktype;
  #else /* _MULTI_THREAD */
    explicit __iar_Lockit(int)
    {   // do nothing
    }

    ~__iar_Lockit()
    {   // do nothing
    }
  #endif /* _MULTI_THREAD */
    __iar_Lockit(const __iar_Lockit&);            // not defined
    __iar_Lockit& operator=(const __iar_Lockit&); // not defined
  };
  class __iar_Lockit_Locale
  {     // lock while object in existence -- MUST NEST
  public:
    explicit __iar_Lockit_Locale() // set the lock
    {
      __iar_Locksyslock_Locale();
    }
    ~__iar_Lockit_Locale()         // clear the lock
    {
      __iar_Unlocksyslock_Locale();
    }
  private:
    __iar_Lockit_Locale(const __iar_Lockit_Locale&);            // not defined
    __iar_Lockit_Locale& operator=(const __iar_Lockit_Locale&); // not defined
  };
  class __iar_Lockit_Malloc
  {     // lock while object in existence -- MUST NEST
  public:
    explicit __iar_Lockit_Malloc() // set the lock
    {
      __iar_Locksyslock_Malloc();
    }
    ~__iar_Lockit_Malloc()         // clear the lock
    {
      __iar_Unlocksyslock_Malloc();
    }
  private:
    __iar_Lockit_Malloc(const __iar_Lockit_Malloc&);            // not defined
    __iar_Lockit_Malloc& operator=(const __iar_Lockit_Malloc&); // not defined
  };
  class __iar_Lockit_Stream
  {     // lock while object in existence -- MUST NEST
  public:
    explicit __iar_Lockit_Stream() // set the lock
    {
      __iar_Locksyslock_Stream();
    }
    ~__iar_Lockit_Stream()         // clear the lock
    {
      __iar_Unlocksyslock_Stream();
    }
  private:
    __iar_Lockit_Stream(const __iar_Lockit_Stream&);            // not defined
    __iar_Lockit_Stream& operator=(const __iar_Lockit_Stream&); // not defined
  };
  class __iar_Lockit_Debug
  {     // lock while object in existence -- MUST NEST
  public:
    explicit __iar_Lockit_Debug() // set the lock
    {
      __iar_Locksyslock_Debug();
    }
    ~__iar_Lockit_Debug()         // clear the lock
    {
      __iar_Unlocksyslock_Debug();
    }
  private:
    __iar_Lockit_Debug(const __iar_Lockit_Debug&);            // not defined
    __iar_Lockit_Debug& operator=(const __iar_Lockit_Debug&); // not defined
  };

  class __iar_Mutex
  {     // lock under program control
  public:
#if _MULTI_THREAD
    __iar_Mutex();
    ~__iar_Mutex();
    void _Lock();
    void _Unlock();
  private:
    __iar_Mutex(const __iar_Mutex&);              // not defined
    __iar_Mutex& operator=(const __iar_Mutex&);   // not defined
    void *__iar_Mtx;
#else /* _MULTI_THREAD */
    void _Lock()
    {   // do nothing
    }
    void _Unlock()
    {   // do nothing
    }
#endif /* _MULTI_THREAD */
  };
_STD_END
#endif /* __cplusplus */

                /* MISCELLANEOUS MACROS AND FUNCTIONS*/
#define _Mbstinit(x)    mbstate_t x = {0, 0}

#define _MAX    max
#define _MIN    min

#if _HAS_NAMESPACE
  #if defined(__cplusplus)
    _STD_BEGIN
    typedef _CSTD __Va_list va_list;
    _STD_END
  #endif /* __cplusplus */
#else
#endif /* _HAS_NAMESPACE */

#endif /* _YVALS */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
