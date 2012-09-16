/* fenv.h standard header */
/* Copyright 2003-2010 IAR Systems AB.  */

#ifndef _FENV
#define _FENV

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#ifndef _YMATH
  #include <ymath.h>
#endif /* _YMATH */

#ifdef _DLIB_PRODUCT_FENV
  #include <DLib_Product_fenv.h>
#else

  _C_STD_BEGIN
  _C_LIB_DECL

                /* TYPES */
  /* The floating-point status flags. */
  typedef unsigned long fexcept_t;
  /* The entire floating-point environment. */
  typedef unsigned long fenv_t;

                /* MACROS */
  /* Supported rounding modes. */
  /* #define FE_DOWNWARD     0x02 */
  #define FE_TONEAREST    0x00
  /* #define FE_TOWARDZERO   0x03 */
  /* #define FE_UPWARD       0x01 */

  /* Supported status flags. */
  /* #define FE_DIVBYZERO    0x02 */
  /* #define FE_INEXACT      0x10 */
  /* #define FE_INVALID      0x01 */
  /* #define FE_OVERFLOW     0x04 */
  /* #define FE_UNDERFLOW    0x08 */
  /* #define FE_ALL_EXCEPT   (  FE_DIVBYZERO | FE_INEXACT \
                            | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW) */
  #define FE_ALL_EXCEPT 0

  _END_C_LIB_DECL
  _C_STD_END
#endif /* _DLIB_PRODUCT_FENV */


_C_STD_BEGIN
_C_LIB_DECL


/* Default floating-point environment. */
#define FE_DFL_ENV      (_CSTD __iar_GetDefaultFenv())

                /* FUNCTION DECLARATIONS */
__ATTRIBUTES int feclearexcept(int);
__ATTRIBUTES int fegetexceptflag(fexcept_t *, int);
__ATTRIBUTES int feraiseexcept(int);
__ATTRIBUTES int fesetexceptflag(const fexcept_t *, int);
__ATTRIBUTES int fetestexcept(int);

__ATTRIBUTES int fegetround(void);
__ATTRIBUTES int fesetround(int);

__ATTRIBUTES int fegetenv(fenv_t *);
__ATTRIBUTES int feholdexcept(fenv_t *);
__ATTRIBUTES int fesetenv(const fenv_t *);
__ATTRIBUTES int feupdateenv(const fenv_t *);

#if _DLIB_ADD_EXTRA_SYMBOLS
  __ATTRIBUTES fexcept_t fegettrapenable(void);
  __ATTRIBUTES int fesettrapenable(fexcept_t);
#endif /* _ADDED_C_LIB */

__ATTRIBUTES fenv_t const *__iar_GetDefaultFenv(void);

_END_C_LIB_DECL
_C_STD_END
#endif /* _FENV */

#if defined(_STD_USING) && defined(__cplusplus)
  using _CSTD fenv_t; using _CSTD fexcept_t;
  using _CSTD feclearexcept; using _CSTD fegetexceptflag;
  using _CSTD feraiseexcept; using _CSTD fesetexceptflag;
  using _CSTD fetestexcept; using _CSTD fegetround;
  using _CSTD fesetround; using _CSTD fegetenv;
  using _CSTD feholdexcept; using _CSTD fesetenv;
  using _CSTD feupdateenv;

  #if _DLIB_ADD_EXTRA_SYMBOLS
    using _CSTD fegettrapenable; using _CSTD fesettrapenable;
  #endif /* _ADDED_C_LIB */

  using _CSTD __iar_GetDefaultFenv;
#endif /* defined(_STD_USING) && defined(__cplusplus) */


/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
