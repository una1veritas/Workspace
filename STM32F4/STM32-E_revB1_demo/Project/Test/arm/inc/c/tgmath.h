/* tgmath.h standard header */
/* Copyright 2009-2010 IAR Systems AB. */
#ifndef _TGMATH
#define _TGMATH

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>

#if _HAS_DINKUM_COMPLEX
  #include <complex.h> 
#endif
#include <math.h>

#ifndef __cplusplus     /* compiler test */
  #define _CARGR(x, fd, ff, fl) \
        __c99_generic(x,,, fd, ff, fl,,,)(x)
  #define _CARGRI(return_type, x, fd, ff, fl) \
        __c99_generic(x,,, fd, ff, fl,,,)(x)
  #define _CARGC(x, cfd, cff, cfl) \
        __c99_generic(x,,, ,,, cfd, cff, cfl)(x)
  #define _CARGCX(x, cfd, cff, cfl) \
        __c99_generic(x,,, ,,, cfd, cff, cfl)(x)
  #define _CARGRC(x, fd, ff, fl, cfd, cff, cfl) \
        __c99_generic(x,,, fd, ff, fl, cfd, cff, cfl)(x)
  #define _CARGRCX(x, fd, ff, fl, cfd, cff, cfl) \
        __c99_generic(x,,, fd, ff, fl, cfd, cff, cfl)(x)
  #define _CARG2R(x, y, fd, ff, fl) \
        __c99_generic(x, y,, fd, ff, fl,,,)(x, y)
  #define _CARG2RX(x, y, fd, ff, fl) \
        __c99_generic(x,,, fd, ff, fl,,,)(x, y)
  #define _CARG2RC(x, y, fd, ff, fl, cfd, cff, cfl) \
        __c99_generic(x, y,, fd, ff, fl, cfd, cff, cfl)(x, y)
  #define _CARG3R(x, y, z, fd, ff, fl) \
        __c99_generic(x, y, z, fd, ff, fl,,,)(x, y, z)
  #define _CARG3RX(x, y, z, fd, ff, fl) \
        __c99_generic(x, y,, fd, ff, fl,,,)(x, y, z)


/* IAR helper macros 
   (suffix annotates nr of params followed by nr of function-names) 
*/
#if _HAS_DINKUM_COMPLEX
  #define _CARGRC_12(x, f1, f2) _CARGRC(x, _D_FUN(f1), _F_FUN(f1), _L_FUN(f1), \
                                           _D_FUN(f2), _F_FUN(f2), _L_FUN(f2))
  #define _CARGRCX_12(x, f1, f2) _CARGRC(x, _D_FUN(f1), _F_FUN(f1), _L_FUN(f1),\
                                            _D_FUN(f2), _F_FUN(f2), _L_FUN(f2))
  #define _CARG2RC_22(x, y, f1, f2) _CARG2RC(x, y, \
                                            _D_FUN(f1), _F_FUN(f1), _L_FUN(f1),\
                                            _D_FUN(f2), _F_FUN(f2), _L_FUN(f2))


  #define _CARGCX_11(x, f)      _CARGCX(x, _D_FUN(f), _F_FUN(f), _L_FUN(f))
  #define _CARGC_11(x, f)       _CARGC(x, _D_FUN(f), _F_FUN(f), _L_FUN(f))
#else /* !_HAS_DINKUM_COMPLEX */
  #define _CARGRC_12(x, f1, f2) _CARGRC(x, _D_FUN(f1), _F_FUN(f1), _L_FUN(f1), \
                                           ,,)
  #define _CARGRCX_12(x, f1, f2) _CARGRC(x, _D_FUN(f1), _F_FUN(f1), _L_FUN(f1),\
                                            ,,)
  #define _CARG2RC_22(x, y, f1, f2) _CARG2RC(x, y, \
                                            _D_FUN(f1), _F_FUN(f1), _L_FUN(f1),\
                                            ,,)
#endif /* _HAS_DINKUM_COMPLEX */

#define _CARG2R_21(x, y, f)     _CARG2R(x, y, _D_FUN(f), _F_FUN(f), _L_FUN(f))
#define _CARG2RX_21(x, y, f)    _CARG2RX(x, y, _D_FUN(f), _F_FUN(f), _L_FUN(f))
#define _CARGR_11(x, f)         _CARGR(x, _D_FUN(f), _F_FUN(f), _L_FUN(f))
#define _CARG3R_31(x, y, z, f)  _CARG3R(x, y, z, \
                                        _D_FUN(f), _F_FUN(f), _L_FUN(f))
#define _CARGRI_21(i, x, f)     _CARGRI(i, x, _D_FUN(f), _F_FUN(f), _L_FUN(f))
#define _CARG3RX_31(x, y, z, f) _CARG3RX(x, y, z, \
                                         _D_FUN(f), _F_FUN(f), _L_FUN(f))




_C_STD_BEGIN
#undef acos
#define acos(x)          _CARGRC_12(x, acos, cacos)
#undef acosh
#define acosh(x)         _CARGRC_12(x, acosh, cacosh)
#undef asin
#define asin(x)          _CARGRC_12(x, asin, casin)
#undef asinh
#define asinh(x)         _CARGRC_12(x, asinh, casinh)
#undef atan
#define atan(x)          _CARGRC_12(x, atan, catan)
#undef atanh
#define atanh(x)         _CARGRC_12(x, atanh, catanh)
#undef atan2
#define atan2(y, x)      _CARG2R_21(y, x, atan2)
#undef carg
#define carg(x)          _CARGCX_11(x, carg)
#undef cbrt
#define cbrt(x)          _CARGR_11(x, cbrt)
#undef ceil
#define ceil(x)          _CARGR_11(x, ceil)
#undef cimag
#define cimag(x)         _CARGCX_11(x, cimag)
#undef conj
#define conj(x)          _CARGC_11(x, conj)
#undef copysign
#define copysign(x, y)   _CARG2R_21(x, y, copysign)
#undef cos
#define cos(x)           _CARGRC_12(x, cos, ccos)
#undef cosh
#define cosh(x)          _CARGRC_12(x, cosh, ccosh)
#undef cproj
#define cproj(x)         _CARGC_11(x, cproj)
#undef creal
#define creal(x)         _CARGCX_11(x, creal)
#if _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH
  #undef erf
  #define erf(x)           _CARGR_11(x, erf)
  #undef erfc
  #define erfc(x)          _CARGR_11(x, erfc)
  #undef expm1
  #define expm1(x)         _CARGR_11(x, expm1)
#endif /* _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH */
#undef exp
#define exp(x)           _CARGRC_12(x, exp, cexp)
#undef exp2
#define exp2(x)          _CARGR_11(x, exp2)
#undef fabs
#define fabs(x)          _CARGRCX_12(x, fabs, cabs)
#undef fdim
#define fdim(x, y)       _CARG2R_21(x, y, fdim)
#undef floor
#define floor(x)         _CARGR_11(x, floor)
#if _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH
  #undef fma
  #define fma(x, y, z)     _CARG3R_31(x, y, z, fma)
#endif /* _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH */
#undef fmax
#define fmax(x, y)       _CARG2R_21(x, y, fmax)
#undef fmin
#define fmin(x, y)       _CARG2R_21(x, y, fmin)
#undef fmod
#define fmod(x, y)       _CARG2R_21(x, y, fmod)
#undef frexp
#define frexp(x, y)      _CARG2RX_21(x, y, frexp)
#undef hypot
#define hypot(x, y)      _CARG2R_21(x, y, hypot)
#undef ilogb
#define ilogb(x)         _CARGRI_21(int, x, ilogb)
#undef ldexp
#define ldexp(x, y)      _CARG2RX_21(x, y, ldexp)
#if _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH
  #undef lgamma
  #define lgamma(x)        _CARGR_11(x, lgamma)
#endif /* _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH */
#undef llrint
#define llrint(x)        _CARGRI_21(_Longlong, x, llrint)
#undef llround
#define llround(x)       _CARGRI_21(_Longlong, x, llround)
#undef log
#define log(x)           _CARGRC_12(x, log, clog)
#undef log1p
#define log1p(x)         _CARGR_11(x, log1p)
#undef log10
#define log10(x)         _CARGR_11(x, log10)
#undef log2
#define log2(x)          _CARGR_11(x, log2)
#undef logb
#define logb(x)          _CARGR_11(x, logb)
#undef lrint
#define lrint(x)         _CARGRI_21(long, x, lrint)
#undef lround
#define lround(x)        _CARGRI_21(long, x, lround)
#undef nearbyint
#define nearbyint(x)     _CARGR_11(x, nearbyint)
#undef nextafter
#define nextafter(x, y)  _CARG2R_21(x, y, nextafter)
#undef nexttoward
#define nexttoward(x, y) _CARG2RX_21(x, y, nexttoward)
#undef pow
#define pow(x, y)        _CARG2RC_22(x, y, pow, cpow)
#undef remainder
#define remainder(x, y)  _CARG2R_21(x, y, remainder)
#undef remquo
#define remquo(x, y, z)  _CARG3RX_31(x, y, z, remquo)
#undef rint
#define rint(x)          _CARGR_11(x, rint)
#undef round
#define round(x)         _CARGR_11(x, round)
#undef scalbln
#define scalbln(x, y)    _CARG2RX_21(x, y, scalbln)
#undef scalbn
#define scalbn(x, y)     _CARG2RX_21(x, y, scalbn)
#undef sin
#define sin(x)           _CARGRC_12(x, sin, csin)
#undef sinh
#define sinh(x)          _CARGRC_12(x, sinh, csinh)
#undef sqrt
#define sqrt(x)          _CARGRC_12(x, sqrt, csqrt)
#undef tan
#define tan(x)           _CARGRC_12(x, tan, ctan)
#undef tanh
#define tanh(x)          _CARGRC_12(x, tanh, ctanh)
#if _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH
  #undef tgamma
  #define tgamma(x)        _CARGR_11(x, tgamma)
#endif /* _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH */
#undef trunc
#define trunc(x)         _CARGR_11(x, trunc)
_C_STD_END

#endif /* __cplusplus */
#endif /* _TGMATH */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
