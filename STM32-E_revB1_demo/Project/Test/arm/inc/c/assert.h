/* assert.h standard header */
/* Copyright 2003-2010 IAR Systems AB. */

/* Note that there is no include guard for this header. This is intentional. */

/*
 * The following symbols control the behaviour of "assert". The
 * default behaviour is to report if the test failed.
 *
 *    NDEBUG   -- The assert expression will not be tested. In fact it will
 *                not even be part of the application, so don't rely on
 *                side effects taking place! (If you create a "Release"
 *                project in Embedded Workbench, this symbol is defined.)
 *
 *    _DLIB_ASSERT_ABORT -- Abort is directly if the test fails.
 *
 *    _DLIB_ASSERT_VERBOSE -- Generate output on the terminal for all
 *                            successful assertions.
 */

/* Note: _VERBOSE_DEBUGGING left in as a synonym for
 * _DLIB_ASSERT_VERBOSE for backward compatibility. */

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>

_C_STD_BEGIN

#undef assert   /* remove existing definition */

#if __AEABI_PORTABILITY_INTERNAL_LEVEL
  #ifndef __AEABI_PORTABLE
    #define __AEABI_PORTABLE
  #endif
#endif /* __AEABI_PORTABILITY_INTERNAL_LEVEL */


_C_LIB_DECL
__ATTRIBUTES void __aeabi_assert(char const *, char const *, int);
__ATTRIBUTES int  __iar_ReportAssert(const char *, const char *,
                                    const char *, const char *);
__ATTRIBUTES void __iar_EmptyStepPoint(void);
__ATTRIBUTES void __iar_PrintAssert(const char*);
_END_C_LIB_DECL

#ifdef NDEBUG

  #define assert(test)  ((void)0)

#else /* NDEBUG */

  #if __AEABI_PORTABILITY_INTERNAL_LEVEL

    #define _STEPPOINT ((void)0)

  #else /* __AEABI_PORTABILITY_INTERNAL_LEVEL */

    /* This allows the debugger to stop on a well-defined point after
     * the assertion. */
    #define _STEPPOINT (_CSTD __iar_EmptyStepPoint())

  #endif /* __AEABI_PORTABILITY_INTERNAL_LEVEL */

  #if   defined(_DLIB_ASSERT_ABORT)
    _C_LIB_DECL
    __ATTRIBUTES_NORETURN void abort(void) _NO_RETURN;
    _END_C_LIB_DECL

    #define assert(test) ((test) ? (void)0 : abort())

  #elif defined(_DLIB_ASSERT_VERBOSE) ||defined(_VERBOSE_DEBUGGING)
    #define _STRIZE(x)    _VAL(x)
    #define _VAL(x)       #x
      #define assert(test) ((test) ? (void)_CSTD __iar_PrintAssert(      \
          __FILE__ ":" _STRIZE(__LINE__) " " #test " -- OK\n")            \
          : (_CSTD __aeabi_assert(#test, __FILE__, __LINE__), _STEPPOINT))

  #else /* _DLIB_ASSERT_VERBOSE */

    #define assert(test) ((test) ? (void)0                              \
        : (_CSTD __aeabi_assert(#test, __FILE__, __LINE__), _STEPPOINT))

  #endif /* _DLIB_ASSERT_XXX */
#endif /* NDEBUG */

_C_STD_END

#ifdef _STD_USING
  using _CSTD __aeabi_assert;
  using _CSTD __iar_ReportAssert;
  using _CSTD __iar_EmptyStepPoint;
#endif /* _STD_USING */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
