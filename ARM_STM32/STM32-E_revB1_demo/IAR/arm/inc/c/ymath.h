/* ymath.h internal header */
/* Copyright 2003-2010 IAR Systems AB. */
#ifndef _YMATH
#define _YMATH

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif


#if !defined(_NO_DEFINITIONS_IN_HEADER_FILES)

#include <ycheck.h>
#include <yvals.h>

_C_STD_BEGIN
_C_LIB_DECL

                /* MACROS FOR _X_FNAME(Dtest) RETURN (0 => ZERO) */
#define _DENORM         (-2)    /* C9X only */
#define _FINITE         (-1)
#define _INFCODE        1
#define _NANCODE        2

                /* TYPE DEFINITIONS */

#if __SHORT_SIZE__ != 2
#error "Float implementation assumes short is 2 bytes"
#endif

typedef union
{       /* pun float types as integer array */
  unsigned short _Word[__LONG_DOUBLE_SIZE__ / 2];
  float _Float;
  double _Double;
  long double _Long_double;
} _Dconst;

                /* double DECLARATIONS */
__EFF_NS __ATTRIBUTES double       _D_FNAME(Cosh)(double, double);
__EFF_NE __ATTRIBUTES short        _D_FNAME(Dtest)(double);
__EFF_NE __ATTRIBUTES int          _D_FNAME(Dsign)(double);
#if _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH
  __EFF_NS __ATTRIBUTES double       _D_FNAME(Erfc)(double);
#endif /* _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH */
__EFF_NS __ATTRIBUTES short        _D_FNAME(Exp)(double *, double, long);
__EFF_NS __ATTRIBUTES double       _D_FNAME(Log)(double, int);
__EFF_NS __ATTRIBUTES double       _D_FNAME(Logpoly)(double);
__EFF_NS __ATTRIBUTES unsigned int _D_FNAME(Quad)(double *);
__EFF_NS __ATTRIBUTES unsigned int _D_FNAME(QuadXp)(double *);
__EFF_NS __ATTRIBUTES unsigned int _D_FNAME(Quadph)(double *, double);
__EFF_NS __ATTRIBUTES double       _D_FNAME(Rint)(double);
__EFF_NS __ATTRIBUTES double       _D_FNAME(Sin)(double, unsigned int);
#ifndef _DLIB_DO_NOT_ADD_SMALL_FUNCTIONS
  __EFF_NS __ATTRIBUTES double _D_FUN(__iar_Sin_small)(double, unsigned int);
#endif /* _DLIB_DO_NOT_ADD_SMALL_FUNCTIONS */
__EFF_NS __ATTRIBUTES double       _D_FNAME(Sinh)(double, double);
__EFF_NS __ATTRIBUTES double       _D_FNAME(Tgamma)(double *, short *pex);
_DLIB_CONST_ATTR extern double const __aeabi_HUGE_VAL;

                /* float DECLARATIONS */
#ifndef _FLOAT_IS_DOUBLE
  __EFF_NS __ATTRIBUTES float        _F_FNAME(Cosh)(float, float);
  __EFF_NE __ATTRIBUTES short        _F_FNAME(Dtest)(float);
  __EFF_NE __ATTRIBUTES int          _F_FNAME(Dsign)(float);
  #if _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH
    __EFF_NS __ATTRIBUTES float        _F_FNAME(Erfc)(float);
  #endif /* _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH */
  __EFF_NS __ATTRIBUTES short        _F_FNAME(Exp)(float *, float, long);
  __EFF_NS __ATTRIBUTES float        _F_FNAME(Log)(float, int);
  __EFF_NS __ATTRIBUTES float        _F_FNAME(Logpoly)(float);
  __EFF_NS __ATTRIBUTES unsigned int _F_FNAME(Quad)(float *);
  __EFF_NS __ATTRIBUTES unsigned int _F_FNAME(QuadXp)(float *);
  __EFF_NS __ATTRIBUTES unsigned int _F_FNAME(Quadph)(float *, float);
  __EFF_NS __ATTRIBUTES float        _F_FNAME(Rint)(float);
  __EFF_NS __ATTRIBUTES float        _F_FNAME(Sin)(float, unsigned int);
  #ifndef _DLIB_DO_NOT_ADD_SMALL_FUNCTIONS
    __EFF_NS __ATTRIBUTES float _F_FUN(__iar_Sin_small)(float, unsigned int);
  #endif /* _DLIB_DO_NOT_ADD_SMALL_FUNCTIONS */
  __EFF_NS __ATTRIBUTES float        _F_FNAME(Sinh)(float, float);
  __EFF_NS __ATTRIBUTES float        _F_FNAME(Tgamma)(float *, short *pex);
#endif /* _FLOAT_IS_DOUBLE */
_DLIB_CONST_ATTR extern float const __aeabi_HUGE_VALF;
_DLIB_CONST_ATTR extern float const __aeabi_INFINITY;
_DLIB_CONST_ATTR extern float const __aeabi_NAN;

                /* long double DECLARATIONS */
#ifndef _LONG_DOUBLE_IS_DOUBLE
  __EFF_NS __ATTRIBUTES long double  _L_FNAME(Cosh)(long double, long double);
  __EFF_NE __ATTRIBUTES short        _L_FNAME(Dtest)(long double);
  __EFF_NE __ATTRIBUTES int          _L_FNAME(Dsign)(long double);
  #if _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH
    __EFF_NS __ATTRIBUTES long double  _L_FNAME(Erfc)(long double);
  #endif /* _DLIB_ALLOW_LARGE_CONSTANT_TABLES_FOR_MATH */
  __EFF_NS __ATTRIBUTES short        _L_FNAME(Exp)(long double *, long double,
                                                   long);
  __EFF_NS __ATTRIBUTES long double  _L_FNAME(Log)(long double, int);
  __EFF_NS __ATTRIBUTES long double  _L_FNAME(Logpoly)(long double);
  __EFF_NS __ATTRIBUTES unsigned int _L_FNAME(Quad)(long double *);
  __EFF_NS __ATTRIBUTES unsigned int _L_FNAME(QuadXp)(long double *);
  __EFF_NS __ATTRIBUTES unsigned int _L_FNAME(Quadph)(long double *, 
                                                      long double);
  __EFF_NS __ATTRIBUTES long double  _L_FNAME(Rint)(long double);
  __EFF_NS __ATTRIBUTES long double  _L_FNAME(Sin)(long double, unsigned int);
  #ifndef _DLIB_DO_NOT_ADD_SMALL_FUNCTIONS
    __EFF_NS __ATTRIBUTES long double _L_FUN(__iar_Sin_small)(long double, 
                                                              unsigned int);
  #endif /* _DLIB_DO_NOT_ADD_SMALL_FUNCTIONS */
  __EFF_NS __ATTRIBUTES long double  _L_FNAME(Sinh)(long double, long double);
  __EFF_NS __ATTRIBUTES long double  _L_FNAME(Tgamma)(long double *, 
                                                      short *pex);
#endif /* _LONG_DOUBLE_IS_DOUBLE */
_DLIB_CONST_ATTR extern long double const __aeabi_HUGE_VALL;

                /* long double ADDITIONS TO math.h NEEDED FOR complex */
__EFF_NS __ATTRIBUTES long double (atan2l)(long double, long double);
__EFF_NS __ATTRIBUTES long double (cosl)(long double);
__EFF_NS __ATTRIBUTES long double (expl)(long double);
__EFF_NS __ATTRIBUTES long double (ldexpl)(long double, int);
__EFF_NS __ATTRIBUTES long double (logl)(long double);
__EFF_NS __ATTRIBUTES long double (powl)(long double, long double);
__EFF_NS __ATTRIBUTES long double (sinl)(long double);
__EFF_NS __ATTRIBUTES long double (sqrtl)(long double);
__EFF_NS __ATTRIBUTES long double (tanl)(long double);
                /* float ADDITIONS TO math.h NEEDED FOR complex */
__EFF_NS __ATTRIBUTES float (atan2f)(float, float);
__EFF_NS __ATTRIBUTES float (cosf)(float);
__EFF_NS __ATTRIBUTES float (expf)(float);
__EFF_NS __ATTRIBUTES float (ldexpf)(float, int);
__EFF_NS __ATTRIBUTES float (logf)(float);
__EFF_NS __ATTRIBUTES float (powf)(float, float);
__EFF_NS __ATTRIBUTES float (sinf)(float);
__EFF_NS __ATTRIBUTES float (sqrtf)(float);
__EFF_NS __ATTRIBUTES float (tanf)(float);
_END_C_LIB_DECL
_C_STD_END

#endif /* !defined(_NO_DEFINITIONS_IN_HEADER_FILES) */

#endif /* _YMATH */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
